using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace User_Interface
{
	public abstract class DaseffectBase
	{
		protected Random _random;

		private double _corruptionRate;

		private float _waterLevel;

		private float _phaseSpeed;

		/// <summary>
		/// Shows that the class instance was successfully initialized.
		/// </summary>
		public bool Ready { get; protected set; }
		
		public const double DefaultCorruptionRate = 0.950;
		
		public const double MinCorruptionRate = 0.001;
		public const double MaxCorruptionRate = 1.000;

		/// <summary>
		/// Shows what percentage of points should be recalculated.
		/// </summary>
		public double CorruptionRate
		{
			get => _corruptionRate;
			
			set
			{
				_corruptionRate = value;

				if(_corruptionRate < MinCorruptionRate)
					_corruptionRate = MinCorruptionRate;

				if(_corruptionRate > MaxCorruptionRate)
					_corruptionRate = MaxCorruptionRate;
			}
		}

		public const float DefaultWaterLevel = 0.5f;
		
		public const float MinWaterLevel = 0.01f;
		public const float MaxWaterLevel = 0.99f;

		/// <summary>
		/// Parameter affecting rendering.
		/// </summary>
		public float WaterLevel
		{
			get => _waterLevel;
			
			set
			{
				_waterLevel = value;

				if(_waterLevel < MinWaterLevel)
					_waterLevel = MinWaterLevel;

				if(_waterLevel > MaxWaterLevel)
					_waterLevel = MaxWaterLevel;
			}
		}

		public const float DefaultPhaseSpeed = 0.495f;
		
		public const float MinPhaseSpeed = 0.01f;
		public const float MaxPhaseSpeed = 0.50f;

		/// <summary>
		/// The Phase Speed of the wave in physical model.
		/// Can be changed at runtime.
		/// </summary>
		public float PhaseSpeed
		{
			get => _phaseSpeed;

			set
			{
				_phaseSpeed = value;

				if(_phaseSpeed < MinPhaseSpeed)
					_phaseSpeed = MinPhaseSpeed;

				if(_phaseSpeed > MaxPhaseSpeed)
					_phaseSpeed = MaxPhaseSpeed;
			}
		}

		public int RandomSeed { get; protected set; }

		/// <summary>
		/// The Width of resulting image.
		/// </summary>
		public int Width { get; protected set; }

		/// <summary>
		/// The Height of resulting image.
		/// </summary>
		public int Height { get; protected set; }

		public abstract bool IsValid();



		public abstract Bitmap GetBitmap();

		/// <summary>
		/// Provide one tick of physical model.
		/// </summary>
		/// <returns></returns>
		public abstract float Iteration(int Ticks);
		
		public abstract void Set(int dim, int x, int y, float value);
	}
}


