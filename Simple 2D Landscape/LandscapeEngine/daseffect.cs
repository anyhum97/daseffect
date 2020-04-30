using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Simple_2D_Landscape.LandscapeEngine
{
	public class daseffect<Type> where Type : struct
	{
		private Type[,] _buffer;

		public int Width { get; private set; }
		public int Height { get; private set; }

		public daseffect()
		{
			Clear();

			if(typeof(Type) != typeof(float)  &&
			   typeof(Type) != typeof(double) &&
			   typeof(Type) != typeof(int))

				throw new Exception("daseffect: UnSupported type Exception");
		}

		public daseffect(int width, int height)
		{
			if(typeof(Type) != typeof(float)  &&
			   typeof(Type) != typeof(double) &&
			   typeof(Type) != typeof(int))

				throw new Exception("daseffect: Unsupported type Exception");

			if(width < 3 || height < 3)
			{
				Clear();
				return;
			}

			Width = width;
			Height = height;

			_buffer = new Type[width, height];
		}

		public bool IsValid()
		{
			if(typeof(Type) != typeof(float)  &&
			   typeof(Type) != typeof(double) &&
			   typeof(Type) != typeof(int))
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

			return true;
		}

		public void Clear()
		{
			_buffer = null;

			Width = 0;
			Height = 0;
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

		public Type Get(int x, int y)
		{
			// This Method Allows to get the Buffer Element
			
			////////////////////////////////////////////////////////////////////////

			x = CoordinateConvertor(x, Width);
			y = CoordinateConvertor(y, Height);

			////////////////////////////////////////////////////////////////////////

			return _buffer[x, y];
		}

		public void Set(int x, int y, Type value)
		{
			// This Method Allows to set the Buffer Element
			
			////////////////////////////////////////////////////////////////////////

			x = CoordinateConvertor(x, Width);
			y = CoordinateConvertor(y, Height);

			////////////////////////////////////////////////////////////////////////
			
			_buffer[x, y] = value;
		}


	}
}
