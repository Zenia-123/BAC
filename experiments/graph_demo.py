import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

class SimpleNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(128,256)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(256,64)
	def forward(self,x):
		x = self.fc1(x)
		x = self.relu(x)
		return self.fc2(x)

model = SimpleNetwork().eval()

graph_model = symbolic_trace(model)

x = torch.randn(1,128)
ShapeProp(graph_model).propagate(x)

graph_model.graph.print_tabular()

print("\n=== 节点与输出形状 ===")
for node in graph_model.graph.nodes:
    meta = node.meta.get("tensor_meta")
    shape = getattr(meta, "shape", None)
    print(node.op, node.target, shape)
    

