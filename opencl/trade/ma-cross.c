
// MA_Cross.Program
__kernel  void maCross(global float* bars, int blength, global float* values, int vlength);

// MA_Cross.Program
__kernel  void maCross(global float* bars, int blength, global float* values, int vlength)
{
	int baseIndex = get_global_id(0);
	int length = baseIndex + 5;
	int position = 0;
  int direction = 0;
	float lengthFloat = (float)length;
  float positionPrice = 0.0;
	float earned = 0.0;
	float ma = 0.0;
	for (int i = 0; i < length; i++) {
		ma += bars[i];
	}
	ma /= lengthFloat;
	position = (bars[length - 1] < ma) ? -1 : ((bars[length - 1] > ma) ? 1 : 0);
	positionPrice = (position != 0) ? bars[length] : 0.0;
	for (int j = length; j < blength - 1; j++) {
		ma += ((bars[j] - bars[j - length]) / lengthFloat);
		direction = (bars[j] < ma) ? -1 : ((bars[j] > ma) ? 1 : 0);
		if (direction != 0 && direction != position) {
			earned += (float)position * (bars[j + 1] - positionPrice);
			positionPrice = bars[j + 1];
			position = direction;
		}
	}
	values[baseIndex] = earned + (position == 0 ? 0.0 :
                                ((float)position * (bars[blength - 1] - positionPrice)));
}
