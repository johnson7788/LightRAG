# Set the device with environment, default is cuda:0
#
# 仓库
# https://github.com/johnson7788/LightRAG

import logging
import os, re
GLM_KEY ="xxxxx"
os.environ["OPENAI_API_KEY"] = GLM_KEY
log_path = os.path.join(os.path.dirname(__file__), "logs")
if not os.path.exists(log_path):
    os.makedirs(log_path)
logfile = os.path.join(log_path, "lightrag.log")
# 日志的格式
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(logfile, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

import datetime
import shutil
from fastapi import FastAPI, UploadFile, Form, File, Request, Query, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse,StreamingResponse,JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing_extensions import Annotated
from typing import List
from enum import Enum
from io import BytesIO
import uvicorn
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding
from lightrag.utils import EmbeddingFunc
import numpy as np
from LightRAG_utils import read_file_content

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LightRAGAPI(object):
    def __init__(self, data_dir="data", upload_dir="upload"):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        self.upload_dir = upload_dir
        if not os.path.exists(self.upload_dir):
            os.mkdir(self.upload_dir)
        # 测试llm和embedding模型
        asyncio.run(self.test_funcs())
        # 初始化1个rag
        self.rag = LightRAG(
            working_dir=self.data_dir,
            llm_model_func=self.llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=768, max_token_size=8192, func=self.embedding_func
            ),
        )

    # function test
    async def test_funcs(self):
        result = await self.llm_model_func("How are you?")
        print("llm_model_func: ", result)

        result = await self.embedding_func(["How are you?"])
        print("embedding_func: ", result)

    async def llm_model_func(self,
            prompt, system_prompt=None, history_messages=[], **kwargs
    ) -> str:
        return await openai_complete_if_cache(
            "glm-4-flash",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            **kwargs,
        )

    async def embedding_func(self, texts: list[str]) -> np.ndarray:
        return await openai_embedding(
            texts,
            model="m3e-base",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="http://127.0.0.1:6303/v1",
        )

    async def rag_qa(self, question, mode='hybrid'):
        """
        进行graphrag问答
        Args:
            question (): text, eg: "What are the top themes in this story?"
            mode (): naive,local,global,hybrid
        Returns:
        """
        if mode not in ["naive","local","global","hybrid"]:
            return False, f"mode {mode}不在预置模式中naive,local,global,hybrid ，请检查"
        rag_result = self.rag.query(question, param=QueryParam(mode="hybrid"))
        return True, rag_result

    async def generate_graph_from_file(self, file):
        """
        上传的文件生成知识图谱
        """
        assert os.path.exists(file), f"文件{file}不存在"
        # 读取文件
        filename = os.path.basename(file)
        content_list = read_file_content(file_path=file)
        # 去掉过多的空内容
        content_list = [content for content in content_list if content]
        content_text = "\n".join(content_list)
        # 生成知识图谱, 增加kv_store_processing.json存储处理完成的文件
        await self.rag.ainsert(content_text,file_names=filename)
        # 根据文件名称获取进度
        logging.info(f"文件{filename}生成知识图谱完成")

    async def get_file_graph_progress(self, filename):
        """
        获取文件生成知识图谱的进度
        """
        result = await self.rag.processing_indicator.get_by_id(id=filename)
        if result == "Done":
            percent = 100
        elif result is None:
            percent = -1
        else:
            percent = 50 # 先写成50，以后改进
        return percent

    async def get_all_graph_progress(self):
        """
        获取所有文件的知识图谱的进度
        """
        result_dict = await light_rag_instance.rag.processing_indicator.get_all()
        result = {}
        for name, value in result_dict.items():
            if value == "Done":
                percent = 100
            elif value is None:
                percent = -1
            else:
                percent = 50  # 先写成50，以后改进
            result[name] = percent
        return result

class Item(BaseModel):
    filename: str

@app.post("/api/genGraph")
async def generate_file_graph(item: Item, background_tasks: BackgroundTasks):
    filename = item.filename
    file_path = os.path.join(light_rag_instance.upload_dir, filename)
    if not os.path.exists(file_path):
        logging.error(f"没有这个文件，File {filename} not found")
        return {"code": 0, "msg": "success", "data": f"File {filename} not found"}
    # 把处理任务放入后台任务队列
    background_tasks.add_task(light_rag_instance.generate_graph_from_file, file_path)
    return {"code": 0, "msg": "success", "data": "请求已收到，正在处理中"}

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>
        </head>
        <body>
            <a href='./docs'>Documents of API</a>
        </body>
    </html>
    """


@app.post("/api/uploads")
async def upload_file_api(file: UploadFile = File(... , description="上传的文件")):
    allowed_extensions = {".pdf", ".txt", ".ppt", ".pptx", ".doc", ".docx", ".xls", ".xlsx"}
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        return {"code": 400, "msg": "上传的文件格式不支持", "data": ""}
    upload_dir = light_rag_instance.upload_dir
    file_path = os.path.join(upload_dir, file.filename)
    if os.path.exists(file_path):
        return False, f"文件{file}已经存在，请删除已有文件后在上传"
    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = {"code": 0, "msg": "success", "data": file_path}
    return result


@app.get("/api/files")
async def list_files():
    """
    获取所有上传的文件
    """
    try:
        if not os.path.exists(light_rag_instance.upload_dir):
            return  {"code": 0, "msg": "success", "data": []}

        files = []
        all_graph_status = await light_rag_instance.get_all_graph_progress()
        for filename in os.listdir(light_rag_instance.upload_dir):
            file_path = os.path.join(light_rag_instance.upload_dir, filename)
            stats = os.stat(file_path)
            files.append({
                "filename": filename,
                "size": stats.st_size,
                "created_at": datetime.datetime.fromtimestamp(stats.st_ctime).isoformat(),
                "graph_status": all_graph_status.get(filename, -1)
            })
        return {"code": 0, "msg": "success", "data": files}
    except Exception as e:
        logging.error(f"列出文件接口报错了，{e}")
        return {"code": 4001, "msg": str(e), "data": []}


@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    try:
        file_path = os.path.join(light_rag_instance.upload_dir, filename)
        if not os.path.exists(file_path):
            logging.error(f"无法删除文件，File {filename} not found")
            return {"code": 0, "msg": "success", "data": f"File {filename} not found"}
        os.remove(file_path)
        return {"code": 0, "msg": "success", "data": f"File {filename} deleted successfully"}
    except Exception as e:
        logging.error(f"删除文件接口报错了，{e}")
        return {"code": 4001, "msg": str(e), "data": []}

@app.get("/api/all_graph_status")
async def get_all_graph_status():
    """
    获取所有文件的graph状态
    """
    result = await light_rag_instance.get_all_graph_progress()
    return {"code": 0, "msg": "success", "data": result}

@app.get("/api/graph_status")
async def get_file_graph_status(filename: str):
    """
    获取指定文件的图谱生成状态
    """
    file_status_percent = await light_rag_instance.get_file_graph_progress(filename)
    return {"code": 0, "msg": "success", "data": file_status_percent}

if __name__ == '__main__':
    light_rag_instance = LightRAGAPI()
    uvicorn.run(app, host='0.0.0.0', port=5307)
