using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Simple_2D_Landscape.LandscapeEngine
{
	public interface IBitmapable
	{
		Bitmap GetBitmap();
	}

	public class Daseffect : IBitmapable, IDisposable
	{
		protected Random _random;

		protected volatile int[] _buffer;

		private double _corruptionRate;

		private float _waterLevel;

		private float _phaseSpeed;

		[DllImport(@"Cuda Implementation.dll")]
		private static extern bool CudaStart(int width, int height);

		[DllImport(@"Cuda Implementation.dll")]
		private static extern void CudaFree();

		[DllImport(@"Cuda Implementation.dll")]
		private static extern bool GetCudaStatus(int width, int height);

		[DllImport(@"Cuda Implementation.dll", CallingConvention = CallingConvention.Cdecl)]
		private static extern bool CudaSetState([In, Out] float[] buffer, int width, int height);

		[DllImport(@"Cuda Implementation.dll")]
		private static extern bool SetDefaultState();

		[DllImport(@"Cuda Implementation.dll")]
		private static extern float CudaCalc(float phaseSpeed);
		
		[DllImport(@"Cuda Implementation.dll", CallingConvention = CallingConvention.Cdecl)]
		private static extern float GetCurrentFrame([In, Out] int[] frame, int ColorInterpretatorIndex, float WaterLevel);

		[DllImport(@"Cuda Implementation.dll", CallingConvention = CallingConvention.Cdecl)]
		private static extern int GetColorInterpretatorTitle([In, Out] StringBuilder str, int ColorInterpretatorIndex);

		[DllImport(@"Cuda Implementation.dll")]
		private static extern int GetColorInterpretatorCount();

		/// <summary>
		/// Shows That the Class Instance was Successfully Initialized.
		/// </summary>
		public bool Ready { get; protected set; }
		
		public const double DefaultCorruptionRate = 0.950;
		
		public const double MinCorruptionRate = 0.001;
		public const double MaxCorruptionRate = 1.000;

		/// <summary>
		/// Shows What Percentage of Points Should be Recalculated.
		/// MinValue => MinCorruptionRate;
		/// MaxValue => MaxCorruptionRate;
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

		public int Width { get; protected set; }
		public int Height { get; protected set; }
		
		public float IterationTime { get; protected set; }
		public float FrameTime { get; protected set; }

		public Daseffect()
		{
			Clear();
		}

		public Daseffect(int width, int height, int seed = 0)
		{
			Ready = false;

			if(width < 3 || height < 3 || height > 1024)
			{
				throw new ArgumentException("> Daseffect: Invalid Field Size");
			}

			if(!CudaStart(width, height))
			{
				return;
			}

			if(!SetDefaultState())
			{
				return;
			}

			if(!GetCudaStatus(width, height))
			{
				return;
			}

			if(seed == 0)
			{
				_random = new Random();
				seed = _random.Next();
				_random = new Random(seed);
				RandomSeed = seed;
			}
			else
			{
				_random = new Random(seed);
				RandomSeed = seed;
			}

			_buffer = new int[width*height];

			CorruptionRate = DefaultCorruptionRate;
			WaterLevel = DefaultWaterLevel;
			PhaseSpeed = DefaultPhaseSpeed;

			Width = width;
			Height = height;

			Ready = true;
		}

		public void Dispose()
		{
			CudaFree();

			GC.SuppressFinalize(this);
		}

		public bool IsValid()
		{
			if(!Ready)
			{
				return false;
			}

			if(_buffer == null)
			{
				return false;
			}

			if(Width < 3 || Height < 3 || Height > 1024)
			{
				return false;
			}

			return GetCudaStatus(Width, Height);
		}

		public void Clear()
		{
			_buffer = null;

			_random = new Random();

			CorruptionRate = DefaultCorruptionRate;
			WaterLevel = DefaultWaterLevel;
			PhaseSpeed = DefaultPhaseSpeed;

			RandomSeed = 0;

			Width = 0;
			Height = 0;

			Ready = false;
		}

		public Bitmap GetBitmap()
		{
			if(!IsValid())
			{
				return null;
			}

			FrameTime = GetFrame();

			if(FrameTime < 0)
			{
				return null;
			}

			Bitmap bitmap = new Bitmap(Width, Height);

			for(int i=0; i<Width; ++i)
			{
				for(int j=0; j<Height; ++j)
				{
					int value = _buffer[i*Height+j];

					Color color = Color.FromArgb((byte)(value >> 24), 
												 (byte)(value >> 16), 
												 (byte)(value >> 8), 
												 (byte)(value));

					bitmap.SetPixel(i, j, color);
				}
			}

			GC.Collect();

			return bitmap;
		}

		public float Iteration()
		{
			if(!IsValid())
			{
				return -1;
			}

			IterationTime = CudaCalc(PhaseSpeed);

			return IterationTime;
		}

		private float GetFrame()
		{
			return GetCurrentFrame(_buffer, 0, WaterLevel);
		}
	}
}
