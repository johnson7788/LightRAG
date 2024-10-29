#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2024/10/29 14:17
# @File  : LightRAG_utils.py
# @Author: 
# @Desc  : pipq install tika

import os
import tika
from tika import parser as tikaParser
TIKA_SERVER_JAR = "file:////media/wac/backup/john/johnson/LightRAG/examples/tika-server.jar"
os.environ['TIKA_SERVER_JAR'] = TIKA_SERVER_JAR

def read_file_content(file_path):
    assert os.path.exists(file_path), f"给定文件不存在: {file_path}"
    tika_jar_path = TIKA_SERVER_JAR.replace('file:///', '')
    assert os.path.exists(tika_jar_path), "tika jar包不存在"
    tika.initVM()
    parsed = tikaParser.from_file(file_path)
    content_text = parsed["content"]
    content = content_text.split("\n")
    return content
