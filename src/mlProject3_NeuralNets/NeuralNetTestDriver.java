package mlProject3_NeuralNets;

import java.util.ArrayList;
import java.util.Random;

public class NeuralNetTestDriver {
	
	public static void main(String[] args) {
		test(44);
	}
	
	private static void test(int iterations) {
		for (int j = 0; j < iterations; j++) {
			Random r = new Random();
			NeuralNetwork nn = new NeuralNetwork(12, 22, 4, 3, false);
			//nn.printAllNodeInfo();
			ArrayList<Double> dubs = new ArrayList<Double>();
			for (int i = 1; i <= 1000; i++) {
				dubs.add(r.nextDouble());
			}
			nn.feedForward(dubs);
		}
	}
}