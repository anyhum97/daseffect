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
		private float[,] _buffer;

		private delegate Color ColorInterpretator(float value, float MinValue, float MaxValue);

		private static readonly ColorInterpretator[] _colorInterpretators;

		private double _corruptionRate = 0.05f;

		private Random _random = new Random();

		private float _bufferMinValue;
		private float _bufferMaxValue;

		private bool ReCount { get; set; }

		private bool Ready { get; set; }

		public enum ColorInterpretatorType
		{
			Default,
			Boolean
		}

		public ColorInterpretatorType _currentColorInterpretator { get; set; } = default;
		
		public const double MinCorruptionRate = 0.005;
		public const double MaxCorruptionRate = 0.995;

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

		public daseffect(int width, int height, int Seed = 0)
		{
			// Min Field is 3x3
			if(width < 3 || height < 3)
			{
				Clear();
				return;
			}

			_buffer = new float[width, height];

			_random = new Random(Seed);

			_bufferMinValue = default;
			_bufferMaxValue = default;

			Width = width;
			Height = height;

			ReCount = true;
			Ready = true;

			Set(Width>>1, Height>>1, 1.0f);
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
			return _colorInterpretators[(int)_currentColorInterpretator](value, _bufferMinValue, _bufferMaxValue);
		}

		private void Count()
		{
			if(ReCount)
			{
				_bufferMinValue = float.MaxValue;
				_bufferMaxValue = float.MinValue;

				for(int i=0; i<Width; ++i)
				{
					for(int j=0; j<Height; ++j)
					{
						float value = _buffer[i, j];

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

			if(_buffer.Length < 9)
			{
				return false;
			}

			if(Width < 3 || Height < 3)
			{
				return false;
			}

			return true;
		}

		public void Clear()
		{
			_buffer = null;

			_currentColorInterpretator = default;

			_bufferMinValue = default;
			_bufferMaxValue = default;
			
			ReCount = true;
			Ready = false;

			Width = 0;
			Height = 0;
		}

		public float Get(int x, int y)
		{
			// This Method Allows to Get the Buffer Element

			////////////////////////////////////////////////////////////////////////

			if(!Ready)
			{
				throw new Exception("> daseffect: Used Invalid Instance");
			}

			////////////////////////////////////////////////////////////////////////

			x = CoordinateConvertor(x, Width);
			y = CoordinateConvertor(y, Height);

			////////////////////////////////////////////////////////////////////////

			return _buffer[x, y];
		}

		public void Set(int x, int y, float value)
		{
			// This Method Allows to Set the Buffer Element
			
			////////////////////////////////////////////////////////////////////////

			if(!Ready)
			{
				throw new Exception("> daseffect: Used Invalid Instance");
			}

			////////////////////////////////////////////////////////////////////////

			x = CoordinateConvertor(x, Width);
			y = CoordinateConvertor(y, Height);

			////////////////////////////////////////////////////////////////////////
			
			_buffer[x, y] = value;
		}

		public Bitmap GetBitmap()
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
					bitmap.SetPixel(i, j, GetDefaultColor(_buffer[i, j], _bufferMinValue, _bufferMaxValue));
				}
			}

			return bitmap;
		}

		public void Iteration()
		{
			// This Methode Performs one Iteration of Physical Calculations
			
			if(!IsValid())
			{
				return;
			}

			// Original Physical Model:

			// Uses the Wave Equation in a Nonequilibrium Medium.
			// Nonequilibrium Medium is Emulated by Random Numbers.

			for(int x=0; x<Width; ++x)
			{
				for(int y=0; y<Height; ++y)
				{
					if(_random.NextDouble() > CorruptionRate)
					{
						float laplacian = Get(x+1, y) + 
										  Get(x-1, y) + 
										  Get(x, y+1) + 
										  Get(x, y-1) - 4.0f * 
										  Get(x, y);

					}
				}
			}

			


		}
	}
}
