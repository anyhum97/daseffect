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
		private float[][][] _buffer;	// [Dimensions][Width][Height];

		private delegate Color ColorInterpretator(float value, float MinValue, float MaxValue);

		private static readonly ColorInterpretator[] _colorInterpretators;

		private Random _random;

		private double _corruptionRate;

		private float _bufferMinValue;
		private float _bufferMaxValue;

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
		
		public const double DefaultCorruptionRate = 1.000;
		
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

		public daseffect(int width, int height, int seed = 0, double corruptionRate = 0.95)
		{
			// Minimus Buffer Size is [3][3][3];

			if(width < 3 || height < 3)
			{
				Clear();
				return;
			}

			_buffer = new float[3][][];

			// Note: First Dimension of Buffer is Responsible for Time
			// We Need Three Time-Shots to Provide Operations with Second Derivative.

			// [0] => t-1	 => past;
			// [1] => t		 => now;
			// [2] => t+1	 => future;

			for(int i=0; i<3; ++i)
			{
				// Allocate [Width] x [Height] Field;

				_buffer[i] = new float[width][];

				for(int j=0; j<width; ++j)
				{
					_buffer[i][j] = new float[height];
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

			Set(0, Width>>1, Height>>1, 1.0f);
			Set(1, Width>>1, Height>>1, 1.0f);
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
						float value = _buffer[dim][i][j];

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

			if(_buffer == null)
			{
				return false;
			}

			if(Width < 3 || Height < 3)
			{
				return false;
			}

			try
			{
				float value = _buffer[2][Width-1][Height-1];
			}
			catch
			{
				return false;
			}

			return true;
		}

		public void Clear()
		{
			_buffer = null;

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

			return _buffer[dim][x][y];
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
			
			_buffer[dim][x][y] = value;
		}

		public Bitmap GetBitmap(int dim = 1)
		{
			// This Methode Returns a Bitmap Image Based on Buffer Elements

			if(!IsValid())
			{
				return null;
			}

			Bitmap bitmap = new Bitmap(Width, Height);

			Count();

			for(int i=0; i<Width; ++i)
			{
				for(int j=0; j<Height; ++j)
				{
					bitmap.SetPixel(i, j, GetDefaultColor(_buffer[dim][i][j], _bufferMinValue, _bufferMaxValue));
				}
			}

			GC.Collect();

			return bitmap;
		}

		public void Iteration()
		{
			// This Methode Performs one Iteration of Physical Calculations
			
			if(!IsValid())
			{
				return;
			}

			//********Original Physical Model********

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

			_bufferMinValue = float.MaxValue;
			_bufferMaxValue = float.MinValue;

			for(int x=0; x<Width; ++x)
			{
				for(int y=0; y<Height; ++y)
				{
					if(_random.NextDouble() <= CorruptionRate || true)
					{
						// Get(0, x, y) => t-1  => past;
						// Get(1, x, y) => t    => now;
						// Get(2, x, y) => t+1  => future;

						float laplacian = Get(1, x + 1, y) +
										  Get(1, x - 1, y) +
										  Get(1, x, y + 1) +
										  Get(1, x, y - 1) - 4.0f *
										  Get(1, x, y);

						float futureState = 1.1f*laplacian + Get(1, x, y) - Get(0, x, y);

						if(futureState < _bufferMinValue)
							_bufferMinValue = futureState;

						if(futureState > _bufferMaxValue)
							_bufferMaxValue = futureState;

						_buffer[2][x][y] = futureState;
					}
				}
			}

			// Push Buffers

			float[][] link = _buffer[0];

			_buffer[0] = _buffer[1];
			_buffer[1] = _buffer[2];
			
			_buffer[2] = link;

			ReCount = true;
		}
	}
}
