package mlProject3_NeuralNets;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

public class Node {
	private ActivationFunction nodeActivationFunction; // the activation function this node will use
	private Random rand = new Random(8); // random generator
	private boolean mutable = true;
	private double input = 0;
	private double output = 0;
	private int inputCounter = 0;
	HashMap<Node, Double> nextLayerNodes = new HashMap<Node, Double>(); // The nodes that are in the next layer after this node. If map is empty, node is output
	ArrayList<Node> prevLayerNodes = new ArrayList<Node>();
	
	public Node(ActivationFunction f) {
		nodeActivationFunction = f;
	}
	
	public void addChild(Node n) {
		addChild(n, rand.nextDouble()*2-1);
	}
	
	public void addChild(Node n, double weight) {
		nextLayerNodes.put(n, weight);
	}
	
	public void changeMutable() {
		mutable = !mutable;
	}
	
	public void changeMutable(boolean mutable) {
		this.mutable = mutable;
	}
	
	public double getOutput() {
		return output;
	}
	
	public void getInput(double singleNodeInput) {
		input += singleNodeInput;
		inputCounter++;
		if (inputCounter == prevLayerNodes.size()) {
			input = nodeActivationFunction.getOutput(input);
			inputCounter = 0;
			output = input;
			input = 0;
			fire();
		}
	}
	
	public void fire() {
		if (nextLayerNodes.size() == 0) {
			// todo: handle output layer
			
			return;
		}
		Node[] nextNodes = (Node[]) nextLayerNodes.keySet().toArray();
		for (int i = 0; i < nextLayerNodes.size(); i++) {
			// sends the weight of the connection to a node in the next layer * the output of the activation function from this layer
			nextNodes[i].getInput(nextLayerNodes.get(nextNodes[i])*output);
		}
	}

}
