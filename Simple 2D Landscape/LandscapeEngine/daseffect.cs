using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Simple_2D_Landscape.LandscapeEngine
{
	public class daseffect<Type> where Type : struct
	{
		private Type[,] _buffer;
		
		private Type [][] buf;

		private delegate Color ColorInterpretator(Type value, Type MinValue, Type MaxValue);	

		private static readonly ColorInterpretator[] _colorInterpretators;

		private bool Ready { get; set; }

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

			if(typeof(Type) != typeof(float)  &&
			   typeof(Type) != typeof(double) &&
			   typeof(Type) != typeof(int))

				throw new Exception("> daseffect: Unsupported type Exception");
		}

		public daseffect(int width, int height)
		{
			if(typeof(Type) != typeof(float)  &&
			   typeof(Type) != typeof(double) &&
			   typeof(Type) != typeof(int))

				throw new Exception("> daseffect: Unsupported type Exception");

			// Min Field is 3x3
			if(width < 3 || height < 3)
			{
				Clear();
				return;
			}

			_buffer = new Type[width, height];

			Width = width;
			Height = height;

			Ready = true;
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

		private static Color GetDefaultColor(Type value, Type MinValue, Type MaxValue)
		{
			// This Method Returns Color Based on Input Value

			return Color.White;
		}
		
		private void Count()
		{
			
		}

		public bool IsValid()
		{
			if(typeof(Type) != typeof(float)  &&
			   typeof(Type) != typeof(double) &&
			   typeof(Type) != typeof(int))
			{
				return false;
			}

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

			//MinValue = default;
			//MaxValue = default;
			
			Ready = false;

			Width = 0;
			Height = 0;
		}

		public Type Get(int x, int y)
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

		public void Set(int x, int y, Type value)
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

			Bitmap bitmap = new Bitmap(Width, Height);

			//Type MinValue = 0;

			for(int i=0; i<Width; ++i)
			{
				for(int j=0; j<Height; ++j)
				{
					//bitmap.SetPixel(i, j, GetDefaultColor(_buffer[x, y]));
				}
			}

			return bitmap;
		}

		public void Iteration()
		{
			// This Methode Performs one Iteration of Physical Calculations




		}
	}
}
