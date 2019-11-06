package mlProject3_NeuralNets;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

// TODO: Auto-generated Javadoc
/**
 * The Class Node.
 */
public class Node {
	
	/** The node activation function. */
	private ActivationFunction nodeActivationFunction; // the activation function this node will use
	
	/** The rand. */
	private Random rand = new Random(); // random generator
	
	/** The mutable. */
	private boolean mutable = true;
	
	/** The input. */
	private double input = 0;
	
	/** The output. */
	private double output = 0;
	
	/** The pre activation function output. */
	private double preActivationFunctionOutput = 0;
	
	/** The input counter. */
	private int inputCounter = 0;
	
	/** The err node. */
	private double partialErrPartialOut = 0;
	
	// Hash map contains the nodes in the next layer and the weights
	/** The next layer nodes. */
	// Use a node in the next layer as a key to get the weight between this node and the node being queried
	HashMap<Node, Double> nextLayerNodes = new HashMap<Node, Double>(); // The nodes that are in the next layer after this node. If map is empty, node is output
	
	/** The next layer nodes update map. */
	// This map is going to be used for batch updating weights after back prop.
	HashMap<Node, Double> nextLayerNodesUpdateMap = new HashMap<Node, Double>(); // The nodes that are in the next layer after this node. If map is empty, node is output
	
	ArrayList<Node> nextLayerNodesArrayList = new ArrayList<Node>();
	
	/** The prev layer nodes. */
	// References to nodes in the previous layer to be used for back prop
	ArrayList<Node> prevLayerNodes = new ArrayList<Node>();
	
	/**
	 * Instantiates a new node.
	 *
	 * @param f the f
	 */
	public Node(ActivationFunction f) {
		nodeActivationFunction = f;
	}
	
	/**
	 * Adds the child.
	 *
	 * @param n the n
	 */
	public void addChild(Node n) {
		addChild(n, (rand.nextDouble()*2-1)*40);
	}
	
	/**
	 * Adds the child.
	 *
	 * @param n the n
	 * @param weight the weight
	 */
	public void addChild(Node n, double weight) {
		nextLayerNodes.put(n, weight);
		nextLayerNodesUpdateMap.put(n, 0.0);
		nextLayerNodesArrayList.add(n);
	}
	
	/**
	 * Adds the parent.
	 *
	 * @param parent the parent
	 */
	public void addParent(Node parent) {
		prevLayerNodes.add(parent);
	}
	
	/**
	 * Sets the err node.
	 *
	 * @param err the new err node
	 */
	public void setPartialErrPartialOut(Double err) {
		partialErrPartialOut = err;
	}
	
	/**
	 * Gets the connection weight.
	 *
	 * @param connectedNode the connected node
	 * @return the connection weight
	 */
	public double getConnectionWeight(Node connectedNode) {
		return nextLayerNodes.get(connectedNode);
	}
	
	public ArrayList<Node> getNextLayerNodes() {
		return nextLayerNodesArrayList;
	}
	
	/**
	 * Gets the err node.
	 *
	 * @return the err node
	 */
	public double getPartialErrPartialOut() {
		return partialErrPartialOut;
	}
	
	/**
	 * Sets a value for the weight between this node and a node in the next layer, n to be updated to.
	 *
	 * @param n A node that is in the layer after this node
	 * @param weight the weight that should be updated after back prop between this node and n (in the next layer)
	 */
	public void addBatchUpdateValue(Node n, double weight) {
		nextLayerNodesUpdateMap.put(n, weight);
	}
	
	/**
	 * Gets the activation function derivative.
	 *
	 * @return the activation function derivative
	 */
	public double getActivationFunctionDerivative() {
		return nodeActivationFunction.getDerivative();
	}
	
	/**
	 * Gets the children.
	 *
	 * @return the children
	 */
	public Node[] getChildren() {
		Node[] nextLayer = new Node[nextLayerNodes.keySet().toArray().length];
		for (int i = 0; i < nextLayer.length; i++) {
			nextLayer[i] = (Node) nextLayerNodes.keySet().toArray()[i];
		}
		return nextLayer;
	}
	
	/**
	 * Change mutable.
	 */
	public void changeMutable() {
		mutable = !mutable;
	}
	
	/**
	 * Change mutable.
	 *
	 * @param mutable the mutable
	 */
	public void changeMutable(boolean mutable) {
		this.mutable = mutable;
	}
	
	/**
	 * Gets the output.
	 *
	 * @return the output
	 */
	public double getOutput() {
		return output;
	}
	
	/**
	 * Gets the pre activation function output.
	 *
	 * @return the pre activation function output
	 */
	public double getPreActivationFunctionOutput() {
		return preActivationFunctionOutput;
	}
	
	/**
	 * Gets the input.
	 *
	 * @param singleNodeInput the single node input
	 * @return the input
	 */
	public void getInput(double singleNodeInput) {
		input += singleNodeInput;
		inputCounter++;
		if (inputCounter >= prevLayerNodes.size()) {
			//System.out.printf("Input = %f%n", input);
			output = nodeActivationFunction.getOutput(input);
			inputCounter = 0;
			preActivationFunctionOutput = input;
			input = 0;
			//System.out.printf("Output = %f%n", output);
			fire();
		}
	}
	
	/**
	 * Fire.
	 */
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
	
	/**
	 * Prints the info.
	 */
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
