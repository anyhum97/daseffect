using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace User_Interface
{
	public partial class Form1 : Form
	{
		private CudaAdaptor daseffect;

		private Bitmap CurrentBitmap { get; set; }

		private Timer _timer;

		private int _counter = 0;

		public const int DefaultFramesPerOperation = 1;

		public const int MinFramesPerOperation = 1;
		public const int MaxFramesPerOperation = 1024;

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

		public Form1()
		{
			InitializeComponent();

			daseffect = new CudaAdaptor(512, 512);

			StartTimer(100);
		}

		private void StartTimer(int interval = 100)
		{
			_timer = new Timer();

			_timer.Interval = interval;
			_timer.Tick += Timer_Tick;

			_timer.Start();
		}

		private void UpdateImage()
		{
			CurrentBitmap = daseffect.GetBitmap();
			pictureBox1.Image = CurrentBitmap;
		}

		private void Timer_Tick(object sender, EventArgs e)
		{
			for(int i=0; i<FramesPerOperation; ++i)
			{
				daseffect.Iteration();
			}

			UpdateImage();

			if(_counter == 16)
			{
				GC.Collect();
				_counter = 0;
			}
			++_counter;
		}
	}
}
