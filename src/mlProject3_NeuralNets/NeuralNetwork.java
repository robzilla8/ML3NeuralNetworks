package mlProject3_NeuralNets;

import java.util.ArrayList;

// TODO: Auto-generated Javadoc
/**
 * The Class NeuralNetwork.
 */
public class NeuralNetwork {
	
	/** The input layer. */
	ArrayList<Node> inputLayer = new ArrayList<Node>();
	
	/** The output layer. */
	ArrayList<Node> outputLayer = new ArrayList<Node>();
	
	/** Eta/Learning Rate */
	private double learningRate = 0;
	/**
	 * Instantiates a new neural network.
	 *
	 * @param inputLayerSize the input layer size
	 * @param hiddenLayerSize the hidden layer size
	 * @param numberOfHiddenLayers the number of hidden layers
	 * @param outputLayerSize the output layer size
	 * @param biasNodes the bias nodes
	 */
	public NeuralNetwork(int inputLayerSize, int hiddenLayerSize, int numberOfHiddenLayers, int outputLayerSize, boolean biasNodes, double learningRate) {
		// set learning rate
		this.learningRate = learningRate;
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
	
	/**
	 * Clone array list.
	 *
	 * @param original the original
	 * @param clone the clone
	 */
	private void cloneArrayList(ArrayList<Node> original, ArrayList<Node> clone) {
		clone.clear();
		for (int i = 0; i < original.size(); i++) {
			clone.add(original.get(i));
		}
	}
	
	/**
	 * Prints the all node info.
	 */
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
	
	/**
	 * Back prop.
	 */
	public void backProp() {
		
	}
	
	/**
	 * Feed forward.
	 *
	 * @param inputs the inputs
	 * @return the array list
	 */
	public ArrayList<Double> feedForward(ArrayList<Double> inputs) {
		System.out.printf("-|-|-|-|Feed Forward-|-|-|-|%n");
		for (int i = 0; i < inputLayer.size(); i++) {
			System.out.printf("	Input is %f to node %d in the input layer%n", inputs.get(i), i);
			inputLayer.get(i).getInput(inputs.get(i));
		}
		ArrayList<Double> outputs = new ArrayList<Double>();
		for (int i = 0; i < outputLayer.size(); i++) {
			Double output = outputLayer.get(i).getOutput();
			outputs.add(output);
		}
		for (double output : outputs) {
			System.out.printf("	Output = %f%n", output);
		}
		return outputs;
	}
	
	/**
	 * Back prop.
	 *
	 * https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
	 * this website really helped and the backprop algorithm here was designed around this walkthrough
	 * 
	 * @param targetValueOfFeatures A vector containing the target values for each of the output layer nodes
	 */
	public void backProp(ArrayList<Double> target) {
		// transform the target value of features into the activation function space
		ArrayList<Double> targetValueOfFeatures = new ArrayList<Double>();
		for (int i = 0; i < target.size(); i++) {
			double input = target.get(i);
			//targetValueOfFeatures.add(Math.pow(Math.E, input)/(Math.pow(Math.E, input)+1)); // Needs to be changed to mesh with activation function
			targetValueOfFeatures.add(input);
			System.out.printf("Target value = %f%n", targetValueOfFeatures.get(i));
		}
		// Iterate through the output layer and find the total error of each output node with respect to 
		for (int i = 0; i < outputLayer.size(); i++) {
			Node curNode = outputLayer.get(i);
			double errCurNode = 0.5*Math.pow(targetValueOfFeatures.get(i) - curNode.getOutput(), 2);
			double partialErrorPartialOut = -1*(targetValueOfFeatures.get(i) - curNode.getOutput());
			curNode.setPartialErrPartialOut(partialErrorPartialOut);
			double partialOutPartialNet = curNode.getActivationFunctionDerivative();
			// Need to adjust each weight calculated to each output node
			for (int j = 0; j < curNode.prevLayerNodes.size(); j++) {
				// The weight of the connection between the current output node and a node in the previous layer
				double partialNetPartialWeight = curNode.prevLayerNodes.get(j).getOutput(); 
				double partialErrorPartialWeight = partialErrorPartialOut * partialOutPartialNet * partialNetPartialWeight;
				double newWeight = partialNetPartialWeight - learningRate * partialErrorPartialWeight;
				// Store the new weight to be updated later
				curNode.prevLayerNodes.get(j).addBatchUpdateValue(curNode, newWeight);
			}
		}
		
		// Now to iterate through the hidden layers down to the input layer
		boolean done = false;
		ArrayList<Node> curLayer = outputLayer.get(0).prevLayerNodes; // The current hidden layer we are working on
		while(!done) {
			// Work backwards through hidden layers until the input layer is reached
			for (Node curNode : curLayer) {
				double partialErrPartialOut = 0;
				
				// Loop calculates the partial error total with respect to this node by summing over partial errors in the next layer
				for (Node nextLayerNode : curNode.getNextLayerNodes()) {
					double partialErrNextNodePartialOutThisNode = nextLayerNode.getPartialErrPartialOut() * nextLayerNode.getActivationFunctionDerivative();
					partialErrPartialOut += partialErrNextNodePartialOutThisNode * curNode.getConnectionWeight(nextLayerNode);
				}
				// Store this value to be used later
				curNode.setPartialErrPartialOut(partialErrPartialOut);
				// Now to calculate partial error of the cur node with respect to the net of current node
				double partialOutPartialNet = curNode.getActivationFunctionDerivative();
				
				// Now to calculate updates to all of the weights going into cur node
				for (Node prevLayerNode : curNode.prevLayerNodes) {
					double partialNetPartialWeight = prevLayerNode.getOutput();
					double partialErrPartialWeight = partialErrPartialOut * partialOutPartialNet * partialNetPartialWeight;
					double weightUpdate = prevLayerNode.getConnectionWeight(curNode) - learningRate * partialErrPartialWeight;
					prevLayerNode.addBatchUpdateValue(curNode, weightUpdate);
				}
			}
			
			// Switch the layer being worked on
			curLayer = curLayer.get(0).prevLayerNodes;
			// Check to see if done
			if (curLayer.get(0).prevLayerNodes.size() == 0) {
				// input layer has a null value for previous layer nodes, so we are done
				done = true;
			}
		}
		
		// Finally update all nodes to their new weight
		curLayer = inputLayer;
		while (curLayer.get(0).getNextLayerNodes().size() != 0) {
			for (Node n : curLayer) {
				n.batchUpdateWeights();
			}
			curLayer = curLayer.get(0).getNextLayerNodes();
		}
		//printAllNodeInfo();
	}
	

}
