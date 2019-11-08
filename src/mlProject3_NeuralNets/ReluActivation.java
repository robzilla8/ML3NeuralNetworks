package mlProject3_NeuralNets;

// TODO: Auto-generated Javadoc
/**
 * The Class ReluActivation.
 */
public class ReluActivation implements ActivationFunction {

	/** Var -- Stores the result of the last value to have been fed through in feed forward after it was mapped to the logisit function */
	private double var = 0;
	
	/**
	 * Gets the output of the RELU function on the input
	 *
	 *	The result of applying this function gets stored in var
	 *
	 * @param input the input to be mapped to the RELU function
	 * @return the output of the input mapped to the RELU function
	 */
	@Override
	public double getOutput(double input) {
		if (input > 0) {
			var = input;
			return input;
		}
		var = 0;
		return 0;
	}

	/**
	 * Gets the derivative.
	 *
	 * @param input the input
	 * @return the derivative
	 */
	@Override
	public double getDerivative(double input) {
		if (input > 0) {
			return 1;
		}
		return 0;
	}

	/**
	 * Gets the derivative of the RELU function at the same point as the last point that was fed
	 * through feed forward and had RELU function applied to it.
	 *
	 * @return the derivative
	 */
	@Override
	public double getDerivative() {
		// TODO Auto-generated method stub
		return getDerivative(var);
	}

}
