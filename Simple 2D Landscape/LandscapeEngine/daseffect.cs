using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Simple_2D_Landscape.LandscapeEngine
{
	public class daseffect
	{
		private float[][][] Buffer { get; set; }	// [Dimensions][Width][Height];

		private delegate Color ColorInterpretator(float value, float MinValue, float MaxValue);

		private static readonly ColorInterpretator[] _colorInterpretators;

		private Random _random;

		private double _corruptionRate;

		private float _bufferMinValue;
		private float _bufferMaxValue;
		private float _bufferSum;

		/// <summary>
		/// Shows Should Metrics be Recalculated;
		/// Metrics: _bufferMinValue, _bufferMaxValue;
		/// </summary>
		private bool ReCount { get; set; }

		/// <summary>
		/// Shows That the Class Instance was Successfully Initialized
		/// And Ready to Work.
		/// </summary>
		private bool Ready { get; set; }

		public enum ColorInterpretatorType
		{
			Default,
			Boolean
		}

		public ColorInterpretatorType CurrentColorInterpretator { get; set; } = default;
		
		public const double DefaultCorruptionRate = 0.900;
		
		public const double MinCorruptionRate = 0.001;
		public const double MaxCorruptionRate = 1.000;

		/// <summary>
		/// Shows What Percentage of Points Should be Recalculated.
		/// MinValue => MinCorruptionRate;
		/// MaxValue => MaxCorruptionRate;
		/// </summary>
		public double CorruptionRate
		{
			get
			{
				return _corruptionRate;
			}
			
			set
			{
				_corruptionRate = value;

				if(_corruptionRate < MinCorruptionRate)
					_corruptionRate = MinCorruptionRate;

				if(_corruptionRate > MaxCorruptionRate)
					_corruptionRate = MaxCorruptionRate;
			}
		}

		public int RandomSeed { get; private set; }

		public int Width { get; private set; }
		public int Height { get; private set; }

		static daseffect()
		{
			_colorInterpretators = new ColorInterpretator[]
			{
				GetDefaultColor
			};
		}

		public daseffect()
		{
			Clear();
		}

		public daseffect(int width, int height, int seed = 0, double corruptionRate = DefaultCorruptionRate)
		{
			// Minimus Buffer Size is [3][3][3];

			if(width < 3 || height < 3)
			{
				Clear();
				return;
			}

			Buffer = new float[3][][];

			// Note: First Dimension of Buffer is Responsible for Time
			// We Need Three Time-Shots to Provide Operations with Second Derivative.

			// [0] => t-1	 => past;
			// [1] => t		 => now;
			// [2] => t+1	 => future;

			for(int i=0; i<3; ++i)
			{
				// Allocate [Width] x [Height] Field;

				Buffer[i] = new float[width][];

				for(int j=0; j<width; ++j)
				{
					Buffer[i][j] = new float[height];
				}
			}

			// We Can Use Seed to Fix the Result
			_random = new Random(seed);

			CorruptionRate = corruptionRate;

			_bufferMinValue = default;
			_bufferMaxValue = default;

			Width = width;
			Height = height;

			ReCount = true;
			Ready = true;
		}

		private static Color GetDefaultColor(float value, float MinValue, float MaxValue)
		{
			// This Method Returns Color Based on Input Value

			if(value == 0.0f)
			{
				return Color.White;
			}

			if(value < 0.0f)
			{
				int intensity = (int)(Math.Floor(255.0f * value / MinValue));
				return Color.FromArgb(0, 0, intensity);
			}
			else
			{
				int intensity = (int)(Math.Floor(255.0f-255.0f * value / MaxValue));
				return Color.FromArgb(intensity, intensity, intensity);
			}
		}
		
		private Color GetColor(float value)
		{
			return _colorInterpretators[(int)CurrentColorInterpretator](value, _bufferMinValue, _bufferMaxValue);
		}

		private static int CoordinateConvertor(int value, int border)
		{
			// This Method Converts Buffer Coordinates in a Certain Way:
			
			// border => 4;

			// In:  [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
			// Out: [ 3,  0,  1,  2,  3,  0,  1,  2,  3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1];
			
			// It helps to close the Buffer.

			////////////////////////////////////////////////////////////////////////

			if(value < 0)
			{
				value = value % border + border;
			}

			if(value >= border)
			{
				value = value % border;
			}

			return value;
		}

		private bool IsHappened()
		{
			return _random.NextDouble() <= CorruptionRate;
		}

		private void Count(int dim = 1)
		{
			if(ReCount)
			{
				_bufferMinValue = float.MaxValue;
				_bufferMaxValue = float.MinValue;

				for(int i=0; i<Width; ++i)
				{
					for(int j=0; j<Height; ++j)
					{
						float value = Buffer[dim][i][j];

						if(value < _bufferMinValue)
						{
							_bufferMinValue = value;
						}

						if(value > _bufferMaxValue)
						{
							_bufferMaxValue = value;
						}
					}
				}

				ReCount = false;
			}
		}

		public bool IsValid()
		{
			if(Ready == false)
			{
				return false;
			}

			if(Buffer == null)
			{
				return false;
			}

			if(Width < 3 || Height < 3)
			{
				return false;
			}

			try
			{
				float value = Buffer[2][Width-1][Height-1];
			}
			catch
			{
				return false;
			}

			return true;
		}

		public void Clear()
		{
			Buffer = null;

			_bufferMinValue = default;
			_bufferMaxValue = default;

			_random = new Random();

			CurrentColorInterpretator = default;
			CorruptionRate = DefaultCorruptionRate;

			ReCount = true;
			Ready = false;

			Width = 0;
			Height = 0;
		}

		public float Get(int dim, int x, int y)
		{
			// This Method Allows to Get the Buffer Element

			////////////////////////////////////////////////////////////////////////

			if(!Ready)
			{
				throw new Exception("> daseffect: Used Invalid Instance");
			}

			if(dim >= 3)
			{
				throw new ArgumentException();
			}

			////////////////////////////////////////////////////////////////////////

			x = CoordinateConvertor(x, Width);
			y = CoordinateConvertor(y, Height);

			////////////////////////////////////////////////////////////////////////

			return Buffer[dim][x][y];
		}

		public void Set(int dim, int x, int y, float value)
		{
			// This Method Allows to Set the Buffer Element
			
			////////////////////////////////////////////////////////////////////////

			if(!Ready)
			{
				throw new Exception("> daseffect: Used Invalid Instance");
			}

			if(dim >= 3)
			{
				throw new ArgumentException();
			}

			////////////////////////////////////////////////////////////////////////

			x = CoordinateConvertor(x, Width);
			y = CoordinateConvertor(y, Height);

			////////////////////////////////////////////////////////////////////////
			
			Buffer[dim][x][y] = value;
		}

		public Bitmap GetBitmap(int dim = 1)
		{
			// This Methode Returns a Bitmap Image Based on Buffer Elements

			if(!IsValid())
				return null;

			Bitmap bitmap = new Bitmap(Width, Height);

			Count();

			for(int i=0; i<Width; ++i)
			{
				for(int j=0; j<Height; ++j)
				{
					bitmap.SetPixel(i, j, GetDefaultColor(Buffer[dim][i][j], _bufferMinValue, _bufferMaxValue));
				}
			}

			GC.Collect();

			return bitmap;
		}

		public void AddNoise(float amplitude, float freq)
		{
			if(!IsValid())
				return;

			if(freq > 1.0f)
				freq = 1.0f;

			if(freq <= 0.0f)
				return;

			for(int i=0; i<Width; ++i)
			{
				for(int j=0; j<Height; ++j)
				{
					if(_random.NextDouble() <= freq)
					{
						Buffer[0][i][j] = amplitude*(float)_random.NextDouble();
					}

					if(_random.NextDouble() <= freq)
					{
						Buffer[1][i][j] = amplitude*(float)_random.NextDouble();
					}
				}
			}
		}

		public void IterationOptimazed()
		{
			// This Methode Performs one Iteration of Physical Calculations
			
			if(!IsValid())
				return;
			
			/////////////////////////////********Original Physical Model********////////////////////////////

			// Uses the Wave Equation in a Nonlinear Physical Environment.
			// Nonlinear Physical Environment is Emulated by Random Numbers.

			// 1. Let u = u(x, y, t);
			// It Means function u Depends of 3 Variables x, y and t.

			// Wave Equation:
			// laplacian(u) = d^2(u)/dt^2;

			// laplacian Definition:
			// laplacian(u) = d^2(u)/dx^2 + d^2(u)/dy^2 in point (x, y, z);

			// Second Derivative Definition:
			// d^2(u(x, y, t))/dx^2 = u(x+1, y, t) - 2*u(x, y, t) + u(x-1, y, t);
			// d^2(u(x, y, t))/dy^2 = u(x, y+1, t) - 2*u(x, y, t) + u(x, y-1, t);
			// d^2(u(x, y, t))/dt^2 = u(x, y, t+1) - 2*u(x, y, t) + u(x, y, t-1);

			// Note: laplacian(u) is not Depend of the Time [t].

			// Re Write Wave Equation:
			// laplacian(u(x, y, t) = d^2(u(x, y, t))/dt^2;
			
			// u(x+1, y, t) - 2*u(x, y, t) + u(x-1, y, t) + u(x, y+1, t) - 2*u(x, y, t) + u(x, y-1, t) = 
			// u(x, y, t+1) - 2*u(x, y, t) + u(x, y, t-1);

			// Combine Together:
			// u(x, y, t+1) = 
			// u(x+1, y, t) + u(x-1, y, t) + u(x, y+1, t) + u(x, y-1, t) - 2*u(x, y, t) - u(x, y, t-1);

			// If We Know u(t-1) State and u(t) State We Can Say what will be u(t+1) [Future State].

			// Now We have Classical Solution of Wave Equation.
			// If We will Update Less Than 100% Points We Will Have an Interesting Picture...

			////////////////////////////////////////////////////////////////////////////////////////////////

			const float velocity = 0.50f;	// Phase Speed;

			// Cycle Optimization Picture:

			// 0##########0
			// #xxxxxxxxxx#
			// #xxxxxxxxxx#
			// #xxxxxxxxxx#
			// #xxxxxxxxxx#
			// 0##########0

			_bufferMinValue = float.MaxValue;
			_bufferMaxValue = float.MinValue;

			_bufferSum = 0.0f;

			// Make Calculation for Top-Left Point:

			if(IsHappened())
			{
				float laplacian =  Buffer[1][1][0] + 
								   Buffer[1][Width-1][0] + 
								   Buffer[1][0][Height-1] + 
								   Buffer[1][0][1] - 4.0f * 
								   Buffer[1][0][0];

				Buffer[2][0][0] = 2.0f*Buffer[1][0][0] - Buffer[0][0][0] + velocity*laplacian;
			}
			else
			{
				Buffer[2][0][0] = Buffer[1][0][0];	// Point Was Not Updated;
			}

			// Make Calculation for Top-Right Point:

			if(IsHappened())
			{
				float laplacian =  Buffer[1][0][0] + 
								   Buffer[1][Width-2][0] + 
								   Buffer[1][Width-1][Height-1] + 
								   Buffer[1][Width-1][1] - 4.0f * 
								   Buffer[1][Width-1][0];

				Buffer[2][Width-1][0] = 2.0f*Buffer[1][Width-1][0] - Buffer[0][Width-1][0] + velocity*laplacian;
			}
			else
			{
				Buffer[2][Width-1][0] = Buffer[1][Width-1][0];	// Point Was Not Updated;
			}

			// Make Calculation for Down-Left Point:
			
			if(IsHappened())
			{
				float laplacian =  Buffer[1][1][Height-1] + 
								   Buffer[1][Width-1][Height-1] + 
								   Buffer[1][0][Height-2] + 
								   Buffer[1][0][0] - 4.0f * 
								   Buffer[1][0][Height-1];

				Buffer[2][0][Height-1] = 2.0f*Buffer[1][0][Height-1] - Buffer[0][0][Height-1] + velocity*laplacian;
			}
			else
			{
				Buffer[2][0][Height-1] = Buffer[1][0][Height-1];	// Point Was Not Updated;
			}

			// Make Calculation for Down-Right Point:

			if(IsHappened())
			{
				float laplacian =  Buffer[1][0][Height-1] + 
								   Buffer[1][Width-2][Height-1] + 
								   Buffer[1][Width-1][Height-2] + 
								   Buffer[1][Width-1][0] - 4.0f * 
								   Buffer[1][Width-1][Height-1];

				Buffer[2][Width-1][Height-1] = 2.0f*Buffer[1][Width-1][Height-1] - Buffer[0][Width-1][Height-1] + velocity*laplacian;
			}
			else
			{
				Buffer[2][Width-1][Height-1] = Buffer[1][Width-1][Height-1];	// Point Was Not Updated;
			}

			_bufferMinValue = Math.Min(_bufferMinValue, Buffer[2][0][0]);
			_bufferMinValue = Math.Min(_bufferMinValue, Buffer[2][Width-1][0]);
			_bufferMinValue = Math.Min(_bufferMinValue, Buffer[2][0][Height-1]);
			_bufferMinValue = Math.Min(_bufferMinValue, Buffer[2][Width-1][Height-1]);

			_bufferMaxValue = Math.Max(_bufferMaxValue, Buffer[2][0][0]);
			_bufferMaxValue = Math.Max(_bufferMaxValue, Buffer[2][Width-1][0]);
			_bufferMaxValue = Math.Max(_bufferMaxValue, Buffer[2][0][Height-1]);
			_bufferMaxValue = Math.Max(_bufferMaxValue, Buffer[2][Width-1][Height-1]);

			_bufferSum += Buffer[2][0][0];
			_bufferSum += Buffer[2][Width-1][0];
			_bufferSum += Buffer[2][0][Height-1];
			_bufferSum += Buffer[2][Width-1][Height-1];
			
			for(int i=1; i<Width-1; ++i)
			{
				// Make Calculation for Top Border:

				if(IsHappened())
				{
					float laplacian = Buffer[1][i+1][0] + 
									   Buffer[1][i-1][0] + 
									   Buffer[1][i][Height-1] + 
									   Buffer[1][i][1] - 4.0f * 
									   Buffer[1][i][0];

					Buffer[2][i][0] = 2.0f*Buffer[1][i][0] - Buffer[0][i][0] + velocity*laplacian;
				}
				else
				{
					Buffer[2][i][0] = Buffer[1][i][0];	// Point Was Not Updated;
				}

				// Make Calculation for Down Border:
				if(IsHappened())
				{
					float laplacian = Buffer[1][i+1][Height-1] + 
									   Buffer[1][i-1][Height-1] + 
									   Buffer[1][i][Height-2] + 
									   Buffer[1][i][0] - 4.0f * 
									   Buffer[1][i][Height-1];

					Buffer[2][i][Height-1] = 2.0f*Buffer[1][i][Height-1] - Buffer[0][i][Height-1] + velocity*laplacian;
				}
				else
				{
					Buffer[2][i][Height-1] = Buffer[1][i][Height-1];	// Point Was Not Updated;
				}

				_bufferMinValue = Math.Min(_bufferMinValue, Buffer[2][i][0]);
				_bufferMinValue = Math.Min(_bufferMinValue, Buffer[2][i][Height-1]);

				_bufferMaxValue = Math.Max(_bufferMaxValue, Buffer[2][i][0]);
				_bufferMaxValue = Math.Max(_bufferMaxValue, Buffer[2][i][Height-1]);

				_bufferSum += Buffer[2][i][0];
				_bufferSum += Buffer[2][i][Height-1];
			}

			for(int i=1; i<Height-1; ++i)
			{
				// Make Calculation for Left Border:

				if(IsHappened())
				{
					float laplacian = Buffer[1][1][i] + 
									   Buffer[1][Width-1][i] + 
									   Buffer[1][0][i-1] + 
									   Buffer[1][0][i+1] - 4.0f * 
									   Buffer[1][0][i];

					Buffer[2][0][i] = 2.0f*Buffer[1][0][i] - Buffer[0][0][i] + velocity*laplacian;
				}
				else
				{
					Buffer[2][0][i] = Buffer[1][0][i];	// Point Was Not Updated;
				}

				// Make Calculation for Right Border:

				if(IsHappened())
				{
					float laplacian =  Buffer[1][0][i] + 
									   Buffer[1][Width-2][i] + 
									   Buffer[1][Width-1][i-1] + 
									   Buffer[1][Width-1][i+1] - 4.0f * 
									   Buffer[1][Width-1][i];

					Buffer[2][Width-1][i] = 2.0f*Buffer[1][Width-1][i] - Buffer[0][Width-1][i] + velocity*laplacian;
				}
				else
				{
					Buffer[2][Width-1][i] = Buffer[1][Width-1][i];	// Point Was Not Updated;
				}

				_bufferMinValue = Math.Min(_bufferMinValue, Buffer[2][0][i]);
				_bufferMinValue = Math.Min(_bufferMinValue, Buffer[2][Width-1][i]);

				_bufferMaxValue = Math.Max(_bufferMaxValue, Buffer[2][0][i]);
				_bufferMaxValue = Math.Max(_bufferMaxValue, Buffer[2][Width-1][i]);

				_bufferSum += Buffer[2][0][i];
				_bufferSum += Buffer[2][Width-1][i];
			}

			for(int i=1; i<Width-1; ++i)
			{
				for(int j=1; j<Height-1; ++j)
				{
					// Make Calculation for Internal Points:
					
					if(IsHappened())
					{
						float laplacian =  Buffer[1][i+1][j] + 
										   Buffer[1][i-1][j] + 
										   Buffer[1][i][j+1] + 
										   Buffer[1][i][j-1] - 4.0f * 
										   Buffer[1][i][j];

						Buffer[2][i][j] = 2.0f*Buffer[1][i][j] - Buffer[0][i][j] + velocity*laplacian;
					}
					else
					{
						Buffer[2][i][j] = Buffer[1][i][j];	// Point Was Not Updated;
					}

					_bufferMinValue = Math.Min(_bufferMinValue, Buffer[2][i][j]);
					_bufferMaxValue = Math.Max(_bufferMaxValue, Buffer[2][i][j]);

					_bufferSum += Buffer[2][i][j];
				}
			}

			// Push Buffers:

			float[][] link = Buffer[0];

			Buffer[0] = Buffer[1];
			Buffer[1] = Buffer[2];
			
			Buffer[2] = link;

			ReCount = false;
		}

		public void Iteration()
		{
			// This Methode Performs one Iteration of Physical Calculations
			
			if(!IsValid())
				return;
			
			/////////////////////////////********Original Physical Model********////////////////////////////

			// Uses the Wave Equation in a Nonlinear Physical Environment.
			// Nonlinear Physical Environment is Emulated by Random Numbers.

			// 1. Let u = u(x, y, t);
			// It Means function u Depends of 3 Variables x, y and t.

			// Wave Equation:
			// laplacian(u) = d^2(u)/dt^2;

			// laplacian Definition:
			// laplacian(u) = d^2(u)/dx^2 + d^2(u)/dy^2 in point (x, y, z);

			// Second Derivative Definition:
			// d^2(u(x, y, t))/dx^2 = u(x+1, y, t) - 2*u(x, y, t) + u(x-1, y, t);
			// d^2(u(x, y, t))/dy^2 = u(x, y+1, t) - 2*u(x, y, t) + u(x, y-1, t);
			// d^2(u(x, y, t))/dt^2 = u(x, y, t+1) - 2*u(x, y, t) + u(x, y, t-1);

			// Note: laplacian(u) is not Depend of the Time [t].

			// Re Write Wave Equation:
			// laplacian(u(x, y, t) = d^2(u(x, y, t))/dt^2;
			
			// u(x+1, y, t) - 2*u(x, y, t) + u(x-1, y, t) + u(x, y+1, t) - 2*u(x, y, t) + u(x, y-1, t) = 
			// u(x, y, t+1) - 2*u(x, y, t) + u(x, y, t-1);

			// Combine Together:
			// u(x, y, t+1) = 
			// u(x+1, y, t) + u(x-1, y, t) + u(x, y+1, t) + u(x, y-1, t) - 2*u(x, y, t) - u(x, y, t-1);

			// If We Know u(t-1) State and u(t) State We Can Say what will be u(t+1) [Future State].

			// Now We have Classical Solution of Wave Equation.
			// If We will Update Less Than 100% Points We Will Have an Interesting Picture...

			////////////////////////////////////////////////////////////////////////////////////////////////

			const float velocity = 0.50f;	// Phase Speed;

			_bufferMinValue = float.MaxValue;
			_bufferMaxValue = float.MinValue;

			_bufferSum = 0.0f;

			for(int i=0; i<Width; ++i)
			{
				for(int j=0; j<Height; ++j)
				{
					// Make Calculation for Internal Points:
					
					if(IsHappened())
					{
						float laplacian =  Get(1, i+1, j) + 
										   Get(1, i-1, j) + 
										   Get(1, i, j+1) + 
										   Get(1, i, j-1) - 4.0f * 
										   Get(1, i, j);

						Buffer[2][i][j] = 2.0f*Buffer[1][i][j] - Buffer[0][i][j] + velocity*laplacian;
					}
					else
					{
						Buffer[2][i][j] = Buffer[1][i][j];	// Point Was Not Updated;
					}

					_bufferMinValue = Math.Min(_bufferMinValue, Buffer[2][i][j]);
					_bufferMaxValue = Math.Max(_bufferMaxValue, Buffer[2][i][j]);

					_bufferSum += Buffer[2][i][j];
				}
			}

			// Push Buffers:

			float[][] link = Buffer[0];

			Buffer[0] = Buffer[1];
			Buffer[1] = Buffer[2];
			
			Buffer[2] = link;

			ReCount = false;
		}
	}
}
