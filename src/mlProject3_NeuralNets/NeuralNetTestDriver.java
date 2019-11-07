package mlProject3_NeuralNets;

import java.util.ArrayList;
import java.util.Random;

public class NeuralNetTestDriver {
	
	public static void main(String[] args) {
		test(1);
	}
	
	private static void test(int iterations) {
		for (int j = 0; j < iterations; j++) {
			Random r = new Random();
			ArrayList<ArrayList<Double>> testData = new ArrayList<ArrayList<Double>>();
			for (int i = 0; i < 400; i++) {
				double randInt1 = (r.nextInt(50) + 50) / 100.0; // random value between 0.5 and 1
				double randInt2 = r.nextInt(50) / 100.0; // random value between 0.0 and 0.5
				double randInt3 = randInt1 - randInt2;
				ArrayList<Double> arr = new ArrayList<Double>();
				arr.add(randInt1);
				arr.add(randInt2);
				arr.add(randInt3);
				testData.add(arr);
			}
			NeuralNetwork nn = new NeuralNetwork(2, 20, 3, 1, false, 0.8);
			nn.printAllNodeInfo();
			ArrayList<Double> dubs = new ArrayList<Double>();
			for (int i = 1; i <= 1000; i++) {
				dubs.add(r.nextDouble());
			}
			for (int i = 0; i < 100000; i++) {
				System.out.printf("BackProp iteration %d%n", i);
				ArrayList<Double> curVector = testData.get(r.nextInt(testData.size()));
				nn.feedForward(curVector);
				ArrayList<Double> target = new ArrayList<Double>();
				target.add(curVector.get(2));
				nn.backProp(target);
				System.out.printf("%n%n%n%n%n");
			}
			
			double pureError = 0;
			for (ArrayList<Double> vector : testData) {
				ArrayList<Double> output = nn.feedForward(vector);
				pureError += Math.abs(output.get(0) - vector.get(2));
				System.out.printf("	Target: %f%n", vector.get(2));
			}
			nn.printAllNodeInfo();
			System.out.printf("Pure error = %f%n", pureError);
			
		}
	}
}