package mlProject3_NeuralNets;

public class ReluActivation implements ActivationFunction {

	private double var = 0;
	@Override
	public double getOutput(double input) {
		if (input > 0) {
			var = input;
			return input;
		}
		var = 0;
		return 0;
	}

	@Override
	public double getDerivative(double input) {
		if (input > 0) {
			return 1;
		}
		return 0;
	}

	@Override
	public double getDerivative() {
		// TODO Auto-generated method stub
		return getDerivative(var);
	}

}
