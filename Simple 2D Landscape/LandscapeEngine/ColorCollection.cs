using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Simple_2D_Landscape.LandscapeEngine
{
	public enum ColorInterpretationType
	{
		Default,
		Boolean,
		Landscape,
		WaterFlow,
	}

	public class ColorCollection
	{
		protected delegate Color ColorInterpretator(float value, float minValue, float maxValue, float waterLevel);

		protected static readonly ColorInterpretator[] _colorInterpretators;

		public ColorInterpretationType CurrentColorInterpretator { get; set; } = default;

		static ColorCollection()
		{
			_colorInterpretators = new ColorInterpretator[]
			{
				GetDefaultColor,
				GetBooleanColor,
				GetLandscapeColor,
				GetWaterFlowColor,
			};
		}

		protected Color GetColor(float value, float minValue, float maxValue, float waterLevel)
		{
			return _colorInterpretators[(int)CurrentColorInterpretator](value, minValue, maxValue, waterLevel);
		}

		private static Color MixColor(Color color1, Color color2, float value, float border1, float border2)
		{
			float distance1 = Math.Abs(border1-value);
			float distance2 = Math.Abs(border2-value);

			float value1 = distance2/(distance1 + distance2);
			float value2 = distance1/(distance1 + distance2);

			return Color.FromArgb((int)(color1.R*value1 + color2.R*value2),
				                  (int)(color1.G*value1 + color2.G*value2),
								  (int)(color1.B*value1 + color2.B*value2));
		}

		private static Color GetDefaultColor(float value, float MinValue, float MaxValue, float waterLevel)
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
		
		private static Color GetBooleanColor(float value, float MinValue, float MaxValue, float waterLevel)
		{
			// This Method Returns Color Based on Input Value

			if(value == 0.0f)
			{
				return Color.White;
			}

			if(value < 0.0f)
			{
				return Color.Blue;
			}

			return Color.Black;
		}

		private static Color GetLandscapeColor(float value, float MinValue, float MaxValue, float waterLevel)
		{
			// This Method Returns Color Based on Input Value

			value = value - MinValue;

			float factor = MaxValue - MinValue;

			if(value < 0.12*factor)
			{
				return Color.FromArgb(6, 0, 47);
			}

			if(value < 0.25*factor)
			{
				return Color.FromArgb(23, 0, 187);
			}

			if(value < 0.33*factor)
			{
				return Color.FromArgb(119, 100, 255);
			}

			if(value < 0.5*factor)
			{
				return Color.FromArgb(119, 100, 255);
			}

			if(value < 0.65*factor)
			{
				return Color.FromArgb(243, 188, 73);
			}

			if(value < 0.80*factor)
			{
				return Color.FromArgb(28, 231, 12);
			}

			if(value < 0.85*factor)
			{
				return Color.FromArgb(32, 210, 180);
			}

			return Color.White;
		}

		private static Color GetWaterFlowColor(float value, float MinValue, float MaxValue, float waterLevel)
		{
			value = value - MinValue;

			float factor = MaxValue - MinValue;

			if(factor == 0.0f)
			{
				return Color.White;
			}

			value /= factor;

			if(value < waterLevel)
			{
				return MixColor(Color.FromArgb(5, 2, 40), Color.FromArgb(162, 249, 240), value, 0.0f, waterLevel);
			}

			//if(value < 0.9f)
			//{
				return MixColor(Color.FromArgb(215, 172, 2), Color.FromArgb(11, 237, 5), value, waterLevel, 0.9f);
			//}

			//return Color.White;
		}


	}
}
