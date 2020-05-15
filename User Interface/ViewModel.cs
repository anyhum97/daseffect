﻿using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;

namespace User_Interface
{
	public partial class Form1
	{
		private Timer _timer;

		private string _lastDirectoryPath = null;

		private int _counter = 0;

		private bool _isRendering = false;

		public const int DefaultFramesPerOperation = 1;

		public const int MinFramesPerOperation = 1;
		public const int MaxFramesPerOperation = 128;

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
