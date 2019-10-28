package mlProject3_NeuralNets;

import java.util.ArrayList;

public class NeuralNetwork {
	ArrayList<Node> inputLayer = new ArrayList<Node>();
	ArrayList<Node> outputLayer = new ArrayList<Node>();
	public NeuralNetwork(int inputLayerSize, int hiddenLayerSize, int numberOfHiddenLayers, int outputLayerSize, boolean biasNodes) {
		// make input layer
		ArrayList<Node> prevLayer = new ArrayList<Node>();
		for (int i = 0; i < inputLayerSize; i++) {
			Node inputLayerNode = new Node(new GaussianActivation());
			inputLayer.add(inputLayerNode);
			prevLayer.add(inputLayerNode);
		}
		// todo: add biasNodes
		
		// make hidden layers
		for (int j = 0; j < numberOfHiddenLayers; j++) {
			ArrayList<Node> currentLayer = new ArrayList<Node>();
			for (int i = 0; i < hiddenLayerSize; i++) {
				Node hiddenLayerNode = new Node(new GaussianActivation());
				currentLayer.add(hiddenLayerNode);
				// add children/parents
				for (int p = 0; p < prevLayer.size(); p++) {
					prevLayer.get(p).addChild(hiddenLayerNode);
					hiddenLayerNode.addParent(prevLayer.get(p));
				}
			}
			cloneArrayList(currentLayer, prevLayer);
		}
		// todo: add bias nodes
		
		// make output layer
		for (int i = 0; i < outputLayerSize; i++) {
			Node outputLayerNode = new Node(new GaussianActivation());
			outputLayer.add(outputLayerNode);
			for (int j = 0; j < prevLayer.size(); j++) {
				prevLayer.get(j).addChild(outputLayerNode);
				outputLayerNode.addParent(prevLayer.get(j));
			}
		}
	}
	
	private void cloneArrayList(ArrayList<Node> original, ArrayList<Node> clone) {
		clone.clear();
		for (int i = 0; i < original.size(); i++) {
			clone.add(original.get(i));
		}
	}
	
	public void printAllNodeInfo() {
		Node[] curLayer = new Node[inputLayer.size()];
		for (int i = 0; i < inputLayer.size(); i++) {
			curLayer[i] = inputLayer.get(i);
		}
		boolean done = false;
		while(!done) {
			for (Node n : curLayer) {
				n.printInfo();
			}
			if (curLayer[0].getChildren().length > 0) {
				curLayer = curLayer[0].getChildren();
			} else {
				done = true;
			}
		}
	}
	
	public void feedForward(ArrayList<Double> inputs) {
		for (int i = 0; i < inputLayer.size(); i++) {
			inputLayer.get(i).getInput(inputs.get(i));
		}
	}

}
