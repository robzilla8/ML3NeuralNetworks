package mlProject3_NeuralNets;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

public class Node {
	private ActivationFunction nodeActivationFunction; // the activation function this node will use
	private Random rand = new Random(); // random generator
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
		addChild(n, (rand.nextDouble()*2-1)*40);
	}
	
	public void addChild(Node n, double weight) {
		nextLayerNodes.put(n, weight);
	}
	
	public void addParent(Node parent) {
		prevLayerNodes.add(parent);
	}
	
	public Node[] getChildren() {
		Node[] nextLayer = new Node[nextLayerNodes.keySet().toArray().length];
		for (int i = 0; i < nextLayer.length; i++) {
			nextLayer[i] = (Node) nextLayerNodes.keySet().toArray()[i];
		}
		return nextLayer;
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
		if (inputCounter >= prevLayerNodes.size()) {
			//System.out.printf("Input = %f%n", input);
			input = nodeActivationFunction.getOutput(input);
			inputCounter = 0;
			output = input;
			input = 0;
			//System.out.printf("Output = %f%n", output);
			fire();
		}
	}
	
	public void fire() {
		if (nextLayerNodes.size() == 0) {
			// todo: handle output layer
			System.out.printf("Output = %f%n", output);
			return;
		}
		Node[] nextNodes = new Node[nextLayerNodes.keySet().toArray().length];
		for (int i = 0; i < nextNodes.length; i++) {
			nextNodes[i] = (Node) nextLayerNodes.keySet().toArray()[i];
		}
		for (int i = 0; i < nextLayerNodes.size(); i++) {
			// sends the weight of the connection to a node in the next layer * the output of the activation function from this layer
			nextNodes[i].getInput(nextLayerNodes.get(nextNodes[i])*output);
		}
	}
	
	public void printInfo() {
		System.out.printf("%n-----Node Values-----%n");
		System.out.printf("	Number of nodes in next layer: %d%n", nextLayerNodes.size());
		System.out.printf("	Number of nodes in prev layer: %d%n", prevLayerNodes.size());
		System.out.printf("	Next Layer Node Weights:%n");
		Node[] nextLayer =  new Node[nextLayerNodes.keySet().toArray().length];
		for (int i = 0; i < nextLayer.length; i++) {
			nextLayer[i] = (Node) nextLayerNodes.keySet().toArray()[i];
		}
		for (int i = 0; i < nextLayerNodes.size(); i++) {
			System.out.printf("		# %f%n", nextLayerNodes.get(nextLayer[i]));
		}
		System.out.println();
	}
}
