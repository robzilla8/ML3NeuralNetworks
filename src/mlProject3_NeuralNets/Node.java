package mlProject3_NeuralNets;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

// TODO: Auto-generated Javadoc
/**
 * The Class Node.
 * Nodes connected together in the right way comprise the neural network, and the neural network class performs various operations on the
 * nodes in order to do calculations with the neural network.
 */
public class Node {
	
	/** The node activation function. */
	private ActivationFunction nodeActivationFunction; // the activation function this node will use
	
	/** The rand. */
	private Random rand = new Random(); // random generator
	
	/** The mutable. Intended to be used to creat bias nodes */
	private boolean mutable = true;
	
	/** The input. */
	private double input = 0;
	
	/** The output. */
	private double output = 0;
	
	/** The pre activation function output. Output before an activation function is applied to this node */
	private double preActivationFunctionOutput = 0;
	
	/** The input counter. Used to count when all nodes in the previous layer have fired and allows this node to fire when a certain threshold has been reached */
	private int inputCounter = 0;
	
	/** The partial derivative of err with respect to the output of this node */
	private double partialErrPartialOut = 0;
	
	/**  Delete latter. */
	ArrayList<Double> inputs = new ArrayList<Double>();
	
	// Hash map contains the nodes in the next layer and the weights
	/** The next layer nodes. */
	// Use a node in the next layer as a key to get the weight between this node and the node being queried
	HashMap<Node, Double> nextLayerNodes = new HashMap<Node, Double>(); // The nodes that are in the next layer after this node. If map is empty, node is output
	
	/** The next layer nodes update map. */
	// This map is going to be used for batch updating weights after back prop.
	HashMap<Node, Double> nextLayerNodesUpdateMap = new HashMap<Node, Double>(); // The nodes that are in the next layer after this node. If map is empty, node is output
	
	/** The next layer nodes array list. */
	ArrayList<Node> nextLayerNodesArrayList = new ArrayList<Node>();
	
	/** The prev layer nodes. */
	// References to nodes in the previous layer to be used for back prop
	ArrayList<Node> prevLayerNodes = new ArrayList<Node>();
	
	/**
	 * Instantiates a new node.
	 *
	 * @param f the activation function that this node will be use
	 */
	public Node(ActivationFunction f) {
		nodeActivationFunction = f;
	}
	
	/**
	 * Adds a child (next layer node) with a random weight in the range [-0.01,0.01]
	 *
	 * @param n the child node to be added
	 */
	public void addChild(Node n) {
		addChild(n, (rand.nextDouble()*0.02 - 0.01));
	}
	
	/**
	 * Adds the child.
	 *
	 * @param n the child node (next layer node) to be added
	 * @param weight the weight
	 */
	public void addChild(Node n, double weight) {
		nextLayerNodes.put(n, weight);
		nextLayerNodesUpdateMap.put(n, 0.0);
		nextLayerNodesArrayList.add(n);
	}
	
	/**
	 * Adds the parent (a node in the previous layer).
	 *
	 * @param parent the parent
	 */
	public void addParent(Node parent) {
		prevLayerNodes.add(parent);
	}
	
	/**
	 * Sets a value for the partial derivative of error with respect to the output of the node.
	 *
	 * @param err the new err node
	 */
	public void setPartialErrPartialOut(Double err) {
		partialErrPartialOut = err;
	}
	
	/**
	 * Gets the connection weight between this node and a node in the next layer
	 *
	 * @param connectedNode the connected node
	 * @return the connection weight
	 */
	public double getConnectionWeight(Node connectedNode) {
		return nextLayerNodes.get(connectedNode);
	}
	
	/**
	 * Gets the next layer nodes.
	 *
	 * @return the next layer nodes
	 */
	public ArrayList<Node> getNextLayerNodes() {
		return nextLayerNodesArrayList;
	}
	
	/**
	 * Gets the error of this node. A partial derivative of the error with respect to the output of this node.
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
	 * Gets the children of this node or nodes in the next layer.
	 *
	 * @return the children of this node (nodes in the next layer)
	 */
	public Node[] getChildren() {
		Node[] nextLayer = new Node[nextLayerNodes.keySet().toArray().length];
		for (int i = 0; i < nextLayer.length; i++) {
			nextLayer[i] = (Node) nextLayerNodes.keySet().toArray()[i];
		}
		return nextLayer;
	}
	
	/**
	 * Change mutable. Whether a node can have its values changed or not
	 * It was intended that this would be used for bias nodes
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
	 * Batch update weights. After an iteration of back prop this method is called and the nodes weights are updated to
	 * the new values determined by back prop
	 */
	public void batchUpdateWeights() {
		Object[] keys = nextLayerNodes.keySet().toArray();
		for (Object key : keys) {
			Node nextLayerNode = (Node) key;
			// System.out.printf("Updating weight from %f to %f%n", nextLayerNodes.get(nextLayerNode), nextLayerNodesUpdateMap.get(nextLayerNode));
			double oldWeight = nextLayerNodes.get(nextLayerNode);
			nextLayerNodes.put(nextLayerNode, nextLayerNodesUpdateMap.get(nextLayerNode));
			double newWeight = nextLayerNodes.get(nextLayerNode);
			double deltaWeight = oldWeight-newWeight;
			System.out.printf("Changed from %f to %f%n", oldWeight, newWeight);
			if (deltaWeight > 0.0) {
				if (nextLayerNodesArrayList.get(0).nextLayerNodes.size() == 0) {
					System.out.print("Hidden Node connected to output layer ");
				} else {
					System.out.print("Node connected to another hidden layer ");
				}
				System.out.printf ("Increased!! Delta weight = %.10f%n", deltaWeight);
			} else if (deltaWeight < 0.0) {
				if (nextLayerNodesArrayList.get(0).nextLayerNodes.size() == 0) {
					System.out.print("Hidden Node connected to output layer ");
				} else {
					System.out.print("Node connected to another hidden layer ");
				}
				System.out.printf("Decreased!! Delta weight = %.10f%n", deltaWeight);
			}
		}
	}
	
	/**
	 * Adds an input
	 * When the same number of inputs have been reached as are connected to this node
	 * it "fires" and sends the nodes in the next layer it's values.
	 *
	 * @param singleNodeInput the single node input (a weight * activation of a node in the previous layer)
	 * @return n/a
	 */
	public void addInput(double singleNodeInput) {
		inputs.add(singleNodeInput);
		input += singleNodeInput;
		inputCounter++;
		if (inputCounter >= prevLayerNodes.size()) {
			//System.out.printf("Input = %f%n", input);
//			if (prevLayerNodes.size() == 0) {
//				// Handle the input layer, no activation function necessary
//				output = input;
//			} else {
				output = nodeActivationFunction.getOutput(input);
//			}
			preActivationFunctionOutput = input;
			input = 0;
			//System.out.printf("Output = %f%n", output);
//			System.out.printf("Firing...input counter = %d%n", inputCounter);
//			for (int i = 0; i < inputs.size(); i++) {
//				System.out.printf("%f,", inputs.get(i));
//			}
//			System.out.println();
			inputs.clear();
			inputCounter = 0;
			fire();
		}
		// test printing for input layer
		if (prevLayerNodes.size() == 0) {
			System.out.printf("	Input layer output = %f%n", output);
		}
	}
	
	/**
	 * Fire. Send the output of this node multiplied by corresponding weights to the nodes in the next layer
	 */
	public void fire() {
		if (nextLayerNodes.size() == 0) {
			// todo: handle output layer
//			System.out.printf("	Output = %f%n", output);
			System.out.printf("	Pre Activation Function Output = %f%n", preActivationFunctionOutput);
			return;
		}
		Node[] nextNodes = new Node[nextLayerNodes.keySet().toArray().length];
		for (int i = 0; i < nextNodes.length; i++) {
			nextNodes[i] = (Node) nextLayerNodes.keySet().toArray()[i];
		}
		for (int i = 0; i < nextLayerNodes.size(); i++) {
			// sends the weight of the connection to a node in the next layer * the output of the activation function from this layer
			nextNodes[i].addInput(nextLayerNodes.get(nextNodes[i])*output);
		}
	}
	
	/**
	 * Prints miscellaneous node info.
	 */
	public void printInfo() {
		String nodeType = "";
		if (nextLayerNodes.size() == 0) {
			nodeType = "Output Node";
		} else if (prevLayerNodes.size() == 0) {
			nodeType = "Input Node";
		} else {
			nodeType = "Hidden Node";
		}
		System.out.printf("%n-----Node Values %s-----%n", nodeType);
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
		System.out.printf("%n	Partial err partial out = %f%n%n", getPartialErrPartialOut());
	}
}
