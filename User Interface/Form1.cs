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
		public Form1()
		{
			InitializeComponent();

			daseffect = new CudaAdaptor(256, 256);
			UpdateImage();

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
			CurrentBitmap = null;

			if(daseffect != null && daseffect.IsValid())
			{
				CurrentBitmap = daseffect.GetBitmap();
			}
			
			pictureBox1.Image = CurrentBitmap;
		}

		private void Timer_Tick(object sender, EventArgs e)
		{
			if(_isRendering)
			{
				if(daseffect != null && daseffect.IsValid())
				{
					daseffect.Iteration(FramesPerOperation);

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

		private void pictureBox1_Click(object sender, EventArgs e)
		{
			_isRendering = !_isRendering;
		}


	}
}
