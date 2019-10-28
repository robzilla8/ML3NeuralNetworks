package mlProject3_NeuralNets;

public class NeuralNetTestDriver {

	public static void main(String[] args) {
		NeuralNetwork nn = new NeuralNetwork(12, 22, 4, 3, false);
		nn.printAllNodeInfo();
	}

}
