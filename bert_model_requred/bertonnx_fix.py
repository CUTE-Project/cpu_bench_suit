import onnx
print("transforming onnx model...")
mp = onnx.load('bertsquad-12.onnx')
input = [i for i in mp.graph.input if i.name == 'unique_ids_raw_output___9:0']
mp.graph.input.remove(input[0])
output = [o for o in mp.graph.output if o.name == 'unique_ids:0']
mp.graph.output.remove(output[0])
node = [n for n in mp.graph.node if n.name == 'unique_ids_graph_outputs_Identity__10']
mp.graph.node.remove(node[0])
print("saving onnx model...")
onnx.save(mp, './bert.onnx')
