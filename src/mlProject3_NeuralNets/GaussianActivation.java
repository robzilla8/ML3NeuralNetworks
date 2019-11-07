package mlProject3_NeuralNets;

public class GaussianActivation implements ActivationFunction {
	private double var;
	@Override
	public double getOutput(double input) {
		// Change the function below, it's just a sigmoid for testing
		var = Math.pow(Math.E, input)/(Math.pow(Math.E, input)+1);
		return var;
	}

	@Override
	public double getDerivative(double input) {
		// TODO Auto-generated method stub
		//System.out.printf("Getting derivative, returning %f%n", input*(1-input));
		return input*(1-input);
	}

	@Override
	public double getDerivative() {
		return getDerivative(var);
	}

}
