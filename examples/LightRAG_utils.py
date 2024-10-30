#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2024/10/29 14:17
# @File  : LightRAG_utils.py
# @Author: 
# @Desc  : pipq install tika

import os
import json
from lightrag.utils import xml_to_json
from neo4j import GraphDatabase
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
