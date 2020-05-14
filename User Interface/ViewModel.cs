using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;

namespace User_Interface
{
	public partial class Form1
	{
		private DaseffectBase daseffect;

		public Bitmap CurrentBitmap { get; set; }

		private Timer _timer;

		private int _counter = 0;

		private bool _isRendering = false;

		public const int DefaultFramesPerOperation = 8;

		public const int MinFramesPerOperation = 1;
		public const int MaxFramesPerOperation = 256;

		private int _framesPerOperation = DefaultFramesPerOperation;

		public int FramesPerOperation
		{
			get => _framesPerOperation;
			
			set
			{
				_framesPerOperation = value;

				if(_framesPerOperation < MinFramesPerOperation)
					_framesPerOperation = MinFramesPerOperation;

				if(_framesPerOperation > MaxFramesPerOperation)
					_framesPerOperation = MaxFramesPerOperation;
			}
		}
	}
}
