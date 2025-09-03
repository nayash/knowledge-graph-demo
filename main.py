import ollama
import sys
import argparse
import json
import pickle
from tqdm import trange
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyvis.network import Network

# Create the parser
parser = argparse.ArgumentParser(description="Script takes a text as input and forms a knowledge graph from it")

# Add arguments
parser.add_argument('--inputpath', type=str, default='./input/metamorphosis-kafka.txt', help='path of file containing text')
parser.add_argument('--outlabel', type=str, required=True, help='name by which output files will be stored')

# Parse the arguments
args = parser.parse_args()
input_file = args.inputpath
out_label = args.outlabel

print(f'script called with input file: {input_file}, output label: {out_label}')

with open("./input/system-prompt-2.txt", "r") as file:
    system_prompt = file.read()

with open(input_file, "r") as file:
    full_content = file.read()

print(f'full content read from {input_file}=>{full_content[:100]}...')

with open("./input/config.json", "r") as file:
    config = json.load(file)

model = config['model_name']

def remove_extras(text):
    return text.strip().replace("\n", "").replace("```", '').replace('json','')

def process_chunk(text, out):
    response = ollama.chat(model=model, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
        ])
    resp_content = remove_extras(response['message']['content'])
    try:
        t = json.loads(resp_content)
        out.extend(t)
    except json.JSONDecodeError:
        print(f"Failed to decode JSON from response: {resp_content}")
    return response

def save_obj(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

all_triples = []

sentences = full_content.split('.')
batch_size = config['batch_size']
for i in tqdm(range(0, len(sentences), batch_size)):
    batch = sentences[i:i+batch_size]
    paragraph = ".".join(batch)
    process_chunk(paragraph, all_triples)
    if i % 100:
        save_obj(all_triples, f"./output/{out_label}_all_triples.pkl")

print(f'{len(all_triples)} triples found')
save_obj(all_triples, f"./output/{out_label}_all_triples.pkl")


G = nx.MultiDiGraph()

# Add nodes and edges
for triple in tqdm(all_triples):
    head = triple['head_entity']['entity']
    if 'tail_entity' in triple:
        tail = triple['tail_entity']['entity']
    else:
        tail = "null"
        print(f'triple with no tail entity: {triple}')
    relation = triple['relation']['relation']
    
    # Add nodes with attributes if available
    G.add_node(head, attr=triple['head_entity']['attribute'])
    G.add_node(tail, attr=triple['tail_entity']['attribute'] if 'tail_entity' in triple and 'attribute' in triple['tail_entity'] else "")
    
    # Add edge with relation label
    G.add_edge(head, tail, relation=relation)

save_obj(G, f"./output/{out_label}_nx_graph.pkl")

net = Network(height='100vh', width='100%', notebook=False)
net.set_edge_smooth('dynamic')  # makes arrows smoother
net.toggle_physics(True)        # allows interactive movement
net.from_nx(G)

template = net.templateEnv.get_template("template.html")
net.template = template

for edge in net.edges:
    edge['label'] = edge['relation']

net.show(f"./output/graph_{out_label}.html")
print(f"Graph saved to ./output/graph_{out_label}.html")