#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2024/10/29 14:17
# @File  : LightRAG_utils.py
# @Author: 
# @Desc  : pipq install tika

import os
import re
import string
import json
import hashlib
import os
import pickle
import asyncio
from functools import wraps
from lightrag.utils import xml_to_json
from neo4j import GraphDatabase
from firecrawl import FirecrawlApp   #pip install firecrawl-py
import fitz  # PyMuPDF
import tika
from tika import parser as tikaParser
TIKA_SERVER_JAR = "file:////media/wac/backup/john/johnson/LightRAG/examples/tika-server.jar"
if not os.path.exists(TIKA_SERVER_JAR):
    TIKA_SERVER_JAR = "file:////Users/admin/git/tika/tika-server-standard-2.9.0-bin/tika-server.jar"
os.environ['TIKA_SERVER_JAR'] = TIKA_SERVER_JAR

class MyFirecrawl():
    def __init__(self, api_key="EXAMPLE", api_url="http://127.0.0.1:3002"):
        """
        :param api_key: api key
        :param api_url: api url, eg: https://api.firecrawl.dev
        """
        self.api_key = api_key
        self.api_url = api_url
        self.app = FirecrawlApp(api_key=api_key, api_url=api_url)

    def craw_website(self, url):
        """
        :param url: website url, eg: https://my-ip.cc/zh-hans
        """
        crawl_status = self.app.crawl_url(
            url,
            params={
                'maxDepth': 1,
                'limit': 100,
                'scrapeOptions': {'formats': ['markdown', 'html']}
            },
            poll_interval=30
        )
        return crawl_status
    def scrape_website(self, url):
        """
        :param url: website url, eg: https://my-ip.cc/zh-hans
        """
        crape_result = self.app.scrape_url(url, params={'formats': ['markdown', 'html']})
        return crape_result

def is_digits_and_punctuation(text):
    # 获取所有数字和标点符号
    allowed_chars = string.digits + string.punctuation

    # 使用正则表达式判断字符串是否只包含这些字符
    pattern = f'^[{re.escape(allowed_chars)}]*$'

    # 使用 fullmatch 来判断整个字符串是否符合规则
    return bool(re.fullmatch(pattern, text))

def read_file_content(file_path):
    assert os.path.exists(file_path), f"给定文件不存在: {file_path}"
    tika_jar_path = TIKA_SERVER_JAR.replace('file:///', '')
    assert os.path.exists(tika_jar_path), "tika jar包不存在"
    tika.initVM()
    parsed = tikaParser.from_file(file_path)
    content_text = parsed["content"]
    content = content_text.split("\n")
    return content

def average_pdf_text_num(pdf_path):
    """
    计算PDF文档所有页面的平均文字数量
    
    Args:
        pdf_path (str): PDF文件路径
    
    Returns:
        float: 平均每页文字数量
    """
    doc = fitz.open(pdf_path)
    total_text_length = 0
    page_count = len(doc)
    
    for page_num in range(page_count):
        page = doc[page_num]
        text = page.get_text()
        total_text_length += len(text)
    
    doc.close()  # 关闭文档
    average_text_num = total_text_length / page_count if page_count > 0 else 0
    return average_text_num

def analyze_pdf_page(pdf_path, page_number):
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]  # 页码从0开始

    # 提取文字
    text = page.get_text()
    text_length = len(text)

    # 提取图片
    images = page.get_images(full=True)
    image_count = len(images)

    doc.close()
    
    return {
        "text_length": text_length,
        "image_count": image_count,
        "is_image_dominant": image_count > 0 and text_length < 100  # 简单判断
    }


def convert_xml_to_json(xml_path):
    """Converts XML file to JSON and saves the output."""
    if not os.path.exists(xml_path):
        print(f"Error: File not found - {xml_path}")
        return None

    json_data = xml_to_json(xml_path)
    if json_data:
        print(f"JSON data: {json_data}")
        return json_data
    else:
        print("Failed to create JSON data")
        return None


def neo4j_process_in_batches(tx, query, data, batch_size):
    """Process data in batches and execute the given query."""
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        tx.run(query, {"nodes": batch} if "nodes" in query else {"edges": batch})


def save_to_neo4j(data_dir, NEO4J_URI="bolt://localhost:7687", NEO4J_USERNAME="neo4j", NEO4J_PASSWORD="neo4j"):
    """
    保存某个目录下的数据到neo4j
    """
    BATCH_SIZE_NODES = 500
    BATCH_SIZE_EDGES = 100
    # Paths
    xml_file = os.path.join(data_dir, "graph_chunk_entity_relation.graphml")
    # Convert XML to JSON
    json_data = convert_xml_to_json(xml_file)
    if json_data is None:
        return

    # Load nodes and edges
    nodes = json_data.get("nodes", [])
    edges = json_data.get("edges", [])

    # Neo4j queries
    create_nodes_query = """
    UNWIND $nodes AS node
    MERGE (e:Entity {id: node.id})
    SET e.entity_type = node.entity_type,
        e.description = node.description,
        e.source_id = node.source_id,
        e.displayName = node.id
    REMOVE e:Entity
    WITH e, node
    CALL apoc.create.addLabels(e, [node.entity_type]) YIELD node AS labeledNode
    RETURN count(*)
    """

    create_edges_query = """
    UNWIND $edges AS edge
    MATCH (source {id: edge.source})
    MATCH (target {id: edge.target})
    WITH source, target, edge,
         CASE
            WHEN edge.keywords CONTAINS 'lead' THEN 'lead'
            WHEN edge.keywords CONTAINS 'participate' THEN 'participate'
            WHEN edge.keywords CONTAINS 'uses' THEN 'uses'
            WHEN edge.keywords CONTAINS 'located' THEN 'located'
            WHEN edge.keywords CONTAINS 'occurs' THEN 'occurs'
           ELSE REPLACE(SPLIT(edge.keywords, ',')[0], '\"', '')
         END AS relType
    CALL apoc.create.relationship(source, relType, {
      weight: edge.weight,
      description: edge.description,
      keywords: edge.keywords,
      source_id: edge.source_id
    }, target) YIELD rel
    RETURN count(*)
    """

    set_displayname_and_labels_query = """
    MATCH (n)
    SET n.displayName = n.id
    WITH n
    CALL apoc.create.setLabels(n, [n.entity_type]) YIELD node
    RETURN count(*)
    """

    # Create a Neo4j driver
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    try:
        # Execute queries in batches
        with driver.session() as session:
            # Insert nodes in batches
            session.execute_write(
                neo4j_process_in_batches, create_nodes_query, nodes, BATCH_SIZE_NODES
            )

            # Insert edges in batches
            session.execute_write(
                neo4j_process_in_batches, create_edges_query, edges, BATCH_SIZE_EDGES
            )

            # Set displayName and labels
            session.run(set_displayname_and_labels_query)

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        driver.close()


def generate_hex_color(word):
    # 使用MD5哈希来确保每个词语生成相同的哈希值
    hash_object = hashlib.md5(word.encode())
    # 取哈希的前6位作为颜色值，确保生成一个#RRGGBB格式的16进制颜色
    hex_color = "#" + hash_object.hexdigest()[:6]
    return hex_color


def url_to_filename(url, content="", max_length=20):
    """
    将url和content转换成文件名称，url中截取10个字符，content中截取10个字符
    content如果包含中文，也需要保留，除了标点符号
    """
    # 每部分的最大长度
    mid_length = max_length // 2

    # 处理 URL 部分
    url = re.sub(r'^https?://', '', url)  # 去掉协议部分
    url_name = re.sub(r'[^a-zA-Z0-9]', '_', url)  # 非字母数字字符替换为下划线

    # 处理 content 部分：保留中文、字母和数字，去除标点
    content_name = re.sub(r'[^\w\u4e00-\u9fff]', '', content)  # 只保留中文、字母、数字

    # 拼接文件名，限制长度
    if content_name:
        file_name = url_name[:mid_length] + "_" + content_name[:mid_length]
    else:
        file_name = url_name[:max_length]
    return file_name

def cal_md5(content):
    content = str(content)
    result = hashlib.md5(content.encode())
    return result.hexdigest()

def async_cache_decorator(func):
    cache_path = "cache"
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)

    @wraps(func)
    async def wrapper(*args, **kwargs):
        usecache = kwargs.get("usecache", True)
        if "usecache" in kwargs:
            del kwargs["usecache"]

        if len(args)> 0:
            if isinstance(args[0],(int, float, str, list, tuple, dict)):
                key = str(args) + str(kwargs) + func.__name__
            else:
                # 第1个参数以后的内容
                key = str(args[1:]) + str(kwargs) + func.__name__
        else:
            key = str(args) + str(kwargs) + func.__name__

        key_file = os.path.join(cache_path, cal_md5(key) + "_cache.pkl")

        if os.path.exists(key_file) and usecache:
            print(f"缓存命中，读取缓存文件: {key_file}")
            try:
                with open(key_file, 'rb') as f:
                    result = pickle.load(f)
                    return result
            except Exception as e:
                print(f"函数{func.__name__}被调用，缓存被命中，读取文件:{key_file}失败，错误信息:{e}")

        # 使用 `await` 调用异步函数
        result = await func(*args, **kwargs)

        if isinstance(result, tuple) and result[0] == False:
            print(f"函数 {func.__name__} 返回结果为 False, 不缓存")
        else:
            with open(key_file, 'wb') as f:
                pickle.dump(result, f)
            print(f"缓存未命中，结果缓存至文件: {key_file}")

        return result

    return wrapper

def cache_decorator(func):
    """
    cache从文件中读取, 当func中存在usecache时，并且为False时，不使用缓存
    Args:
        func ():
    Returns:
    """
    cache_path = "cache" #cache目录
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)

    @wraps(func)
    def wrapper(*args, **kwargs):
        # 将args和kwargs转换为哈希键， 当装饰类中的函数的时候，args的第一个参数是实例化的类，这会通常导致改变，我们不想检测它是否改变，那么就忽略它
        usecache = kwargs.get("usecache", True)
        if "usecache" in kwargs:
            del kwargs["usecache"]
        if len(args)> 0:
            if isinstance(args[0],(int, float, str, list, tuple, dict)):
                key = str(args) + str(kwargs) + func.__name__
            else:
                # 第1个参数以后的内容
                key = str(args[1:]) + str(kwargs) + func.__name__
        else:
            key = str(args) + str(kwargs) + func.__name__
        # 变成md5字符串
        key_file = os.path.join(cache_path, cal_md5(key) + "_cache.pkl")
        # 如果结果已缓存，则返回缓存的结果
        if os.path.exists(key_file) and usecache:
            # 去掉kwargs中的usecache
            print(f"函数{func.__name__}被调用，缓存被命中，使用已缓存结果，对于参数{key}, 读取文件:{key_file}")
            try:
                with open(key_file, 'rb') as f:
                    result = pickle.load(f)
                    return result
            except Exception as e:
                print(f"函数{func.__name__}被调用，缓存被命中，读取文件:{key_file}失败，错误信息:{e}")
        result = func(*args, **kwargs)
        # 将结果缓存到文件中
        # 如果返回的数据是一个元祖，并且第1个参数是False,说明这个函数报错了，那么就不缓存了，这是我们自己的一个设定
        if isinstance(result, tuple) and result[0] == False:
            print(f"函数{func.__name__}被调用，返回结果为False，对于参数{key}, 不缓存")
        else:
            with open(key_file, 'wb') as f:
                pickle.dump(result, f)
            print(f"函数{func.__name__}被调用，缓存未命中，结果被缓存，对于参数{key}, 写入文件:{key_file}")
        return result

    return wrapper