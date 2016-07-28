#pragma OPENCL EXTENSION cl_khr_fp64: enable

// MA_Cross.Program
__kernel  void maCross(global double* bars, int blength, global int* values, int vlength);

// MA_Cross.Program
__kernel  void maCross(global double* bars, int blength, global int* values, int vlength)
{
	int baseIndex = get_global_id(0);
  if (baseIndex > vlength - 1) {
    return;
  }
	int length = baseIndex + 5;
	int position = 0;
  int direction = 0;
	double lengthDouble = (double)length;
  double positionPrice = 0.0;
	double earned = 0.0;
	double ma = 0.0;
	for (int i = 0; i < length; i++) {
		ma += bars[i];
	}
	ma /= lengthDouble;
	position = (bars[length - 1] < ma) ? -1 : ((bars[length - 1] > ma) ? 1 : 0);
	positionPrice = (position != 0) ? bars[length] : 0.0;
	for (int j = length; j < blength - 1; j++) {
		ma += ((bars[j] - bars[j - length]) / lengthDouble);
		direction = (bars[j] < ma) ? -1 : ((bars[j] > ma) ? 1 : 0);
		if (direction != 0 && direction != position) {
			earned += (double)position * (bars[j + 1] - positionPrice);
			positionPrice = bars[j + 1];
			position = direction;
		}
	}
  // earned += (position == 0 ? 0.0 :
  //            ((double)position * (bars[blength - 1] - positionPrice)));
  if (position > 0) {
    int value = (int)1000 + (int)((bars[blength - 1] - positionPrice) * 100);
    values[baseIndex] = value;//;
  } else if (position < 0) {
    int value = (int)1000 + (int)((bars[blength - 1] - positionPrice) * -100);
    values[baseIndex] = value;//positionPrice - bars[blength - 1];
  } else {
    values[baseIndex] = 0;
  }
  // values[baseIndex] = (int)(earned * 100);
	// values[baseIndex] = earned + (position == 0 ? 0.0 :
  //                               ((double)position * (bars[blength - 1] - positionPrice)));
}
