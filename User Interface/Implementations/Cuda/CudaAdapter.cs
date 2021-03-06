﻿using System;
using System.Collections.Generic;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows.Forms;

namespace User_Interface
{
	public sealed class CudaAdapter : DaseffectBase
	{
		/// <summary>
		/// Used in frame transactions.
		/// [Width x Height];
		/// </summary>
		private volatile int[] _buffer;

		/// <summary>
		/// Used to send current field state.
		/// [2 x Width x Height];
		/// </summary>
		private volatile float[] _send;

		/// <summary>
		/// Import dll functions to use the Cuda Implementation.
		/// </summary>
		
		[DllImport(@"Cuda Implementation.dll")]
		private static extern bool CudaStart(int width, int height);

		[DllImport(@"Cuda Implementation.dll")]
		private static extern void CudaFree();

		[DllImport(@"Cuda Implementation.dll")]
		private static extern bool GetCudaStatus(int width, int height);

		[DllImport(@"Cuda Implementation.dll", CallingConvention = CallingConvention.Cdecl)]
		private static extern bool CudaGetState([In, Out] float[] buffer, int width, int height);

		[DllImport(@"Cuda Implementation.dll", CallingConvention = CallingConvention.Cdecl)]
		private static extern bool CudaSetState([In, Out] float[] buffer, int width, int height);

		[DllImport(@"Cuda Implementation.dll")]
		private static extern bool SetDefaultState();

		[DllImport(@"Cuda Implementation.dll")]
		private static extern float CudaCalc(float phaseSpeed, int ticks);
		
		[DllImport(@"Cuda Implementation.dll", CallingConvention = CallingConvention.Cdecl)]
		private static extern float GetCurrentFrame([In, Out] int[] frame, int ColorInterpretatorIndex, float WaterLevel);

		[DllImport(@"Cuda Implementation.dll", CallingConvention = CallingConvention.Cdecl)]
		private static extern int GetColorInterpretatorTitle([In, Out] StringBuilder str, int ColorInterpretatorIndex);

		[DllImport(@"Cuda Implementation.dll")]
		private static extern int GetColorInterpretatorCount();
		
		public float IterationTime { get; private set; }
		public float FrameTime { get; private set; }

		public CudaAdapter()
		{
			Clear();
		}

		public CudaAdapter(int width, int height, int seed = 0)
		{
			Ready = false;

			if(width < 3 || height < 3 || height > 1024)
			{
				throw new ArgumentException("> Daseffect: Invalid Field Size");
			}

			string errorMessage = "Unable to start the GPU version.\nCheck if you have a suitable device";

			if(!CudaStart(width, height))
			{
				MessageBox.Show(errorMessage);
				return;
			}

			if(!SetDefaultState())
			{
				MessageBox.Show(errorMessage);
				return;
			}

			if(!GetCudaStatus(width, height))
			{
				MessageBox.Show(errorMessage);
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

			_send = new float[2*width*height];

			ColorInterpretatorCount = GetColorInterpretatorCount();

			CorruptionRate = DefaultCorruptionRate;
			WaterLevel = DefaultWaterLevel;
			PhaseSpeed = DefaultPhaseSpeed;

			Width = width;
			Height = height;

			Ready = true;
		}

		public override void Dispose()
		{
			CudaFree();
		}

		public override bool IsValid()
		{
			if(!Ready)
			{
				return false;
			}

			if(_buffer == null || _send == null)
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
			_send = null;

			_random = new Random();

			CorruptionRate = DefaultCorruptionRate;
			WaterLevel = DefaultWaterLevel;
			PhaseSpeed = DefaultPhaseSpeed;
			ColorInterpretatorIndex = 0;

			RandomSeed = 0;
			IterationCount = 0;

			Width = 0;
			Height = 0;

			Ready = false;
		}

		public override void AddNoise(float minValue, float maxValue, float freq)
		{
			if(!IsValid())
			{
				return;
			}

			if(freq > 1.0f)
			{
				freq = 1.0f;
			}

			if(freq <= 0.0f)
			{
				return;
			}

			for(int i=0; i<Width; ++i)
			{
				for(int j=0; j<Height; ++j)
				{
					if(_random.NextDouble() <= freq)
					{
						float value = minValue + (maxValue-minValue)*(float)(_random.NextDouble());
						
						_send[i*Height+j] = value;
					}

					if(_random.NextDouble() <= freq)
					{
						float value = minValue + (maxValue-minValue)*(float)(_random.NextDouble());
						
						_send[Width*Height + i*Height+j] = value;
					}
				}
			}

			CudaSetState(_send, Width, Height);
		}

		public override void AddNoise(float value, float freq)
		{
			AddNoise(value, value, freq);
		}

		public override Bitmap GetBitmap()
		{
			if(!IsValid())
			{
				return null;
			}

			FrameTime = GetCurrentFrame(_buffer, ColorInterpretatorIndex, WaterLevel);

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

			return bitmap;
		}

		public override float Iteration(int ticks)
		{
			if(!IsValid())
			{
				return -1.0f;
			}

			if(ticks < 1)
			{
				ticks = 1;
			}

			IterationCount += ticks;

			return IterationTime = CudaCalc(PhaseSpeed, ticks);
		}

		public override List<string> GetColorInterpretatorsTitle()
		{
			List<string> list = new List<string>();

			if(!IsValid())
			{
				return list;
			}

			int count = GetColorInterpretatorCount();

			for(int i=0; i<count; ++i)
			{
				StringBuilder stringBuilder = new StringBuilder(64);

				int len = GetColorInterpretatorTitle(stringBuilder, i);

				list.Add(stringBuilder.ToString(0, len));
			}

			return list;
		}

		public override void SetColorInterpretator(int index)
		{
			if(!IsValid())
			{
				return;
			}

			if(index >= ColorInterpretatorCount)
			{
				index = 0;
			}

			ColorInterpretatorIndex = index;
		}
	}
}



