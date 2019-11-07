package mlProject3_NeuralNets;

public class LogisticFunction implements ActivationFunction {

	private double var;
	@Override
	public double getOutput(double input) {
		// Change the function below, it's just a sigmoid for testing
		var = 1/(Math.pow(Math.E, -1*input)+1);
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
