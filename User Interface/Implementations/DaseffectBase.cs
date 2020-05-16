using System;
using System.Collections.Generic;
using System.Drawing;

namespace User_Interface
{
	public abstract class DaseffectBase : IDisposable
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
		
		public const double MinCorruptionRate = 0.01;
		public const double MaxCorruptionRate = 1.00;

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
				{
					_corruptionRate = MinCorruptionRate;
				}

				if(_corruptionRate > MaxCorruptionRate)
				{
					_corruptionRate = MaxCorruptionRate;
				}
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
				{
					_waterLevel = MinWaterLevel;
				}

				if(_waterLevel > MaxWaterLevel)
				{
					_waterLevel = MaxWaterLevel;
				}
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
				{
					_phaseSpeed = MinPhaseSpeed;
				}

				if(_phaseSpeed > MaxPhaseSpeed)
				{
					_phaseSpeed = MaxPhaseSpeed;
				}
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
		
		public int IterationCount { get; protected set; } = default;

		public int ColorInterpretatorCount { get; protected set; }

		public int ColorInterpretatorIndex { get; protected set; } = default;

		public abstract bool IsValid();

		/// <summary>
		/// Adds Random Noise (Using Random Seed).
		/// </summary>
		public abstract void AddNoise(float minValue, float maxValue, float freq);

		public abstract void AddNoise(float value, float freq);

		public abstract Bitmap GetBitmap();

		/// <summary>
		/// Provide one tick of physical model.
		/// </summary>
		/// <returns></returns>
		public abstract float Iteration(int ticks);

		public abstract List<string> GetColorInterpretatorsTitle();

		public abstract void SetColorInterpretator(int index);

		public abstract void Dispose();
	}
}


