package mlProject3_NeuralNets;

// TODO: Auto-generated Javadoc
/**
 * The Class LinearActivation.
 * This activation function is used on the input and output layers
 * and simply returns whatever it was passed in. This was necessary as node
 * has to have an activation function the way the program was built
 */
public class LinearActivation implements ActivationFunction {

	/**
	 * Gets the output.
	 *
	 * @param input the input
	 * @return the output
	 */
	@Override
	public double getOutput(double input) {
		// TODO Auto-generated method stub
		return input;
	}

	/**
	 * Gets the derivative.
	 *
	 *	Since this is a linear function, the derivative is simply some constant (but not 0). It was decided that 1 would be used.
	 * @param input the input
	 * @return 1
	 */
	@Override
	public double getDerivative(double input) {
		// TODO Auto-generated method stub
		return 1;
	}

	/**
	 * Gets the derivative.
	 *
	 * @return the derivative
	 */
	@Override
	public double getDerivative() {
		// TODO Auto-generated method stub
		return 1;
	}

}
