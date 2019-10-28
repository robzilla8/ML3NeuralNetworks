package mlProject3_NeuralNets;

public class GaussianActivation implements ActivationFunction {

	@Override
	public double getOutput(double input) {
		// Change the function below, it's just a sigmoid for testing
		return Math.pow(Math.E, input)/(Math.pow(Math.E, input)+1);
	}

	@Override
	public double getGradientt(double input) {
		// TODO Auto-generated method stub
		return 0;
	}

}
