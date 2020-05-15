using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace User_Interface
{
	public partial class Form1 : Form
	{
		private DaseffectBase daseffect;

		public Bitmap CurrentBitmap { get; set; }

		public Form1()
		{
			InitializeComponent();
			InitializeViewModel();
			StartTimer(80);
		}

		private void InitializePhysicalModel(int modelType = 0)
		{
			bool saveState = _isRendering;

			_isRendering = false;

			if(modelType == 0)
			{
				daseffect = new Daseffect(512, 512);
			}
			else
			{
				daseffect = new CudaAdaptor(512, 512);
			}

			daseffect.AddNoise(0.001f, 0.1f, 0.005f);

			_isRendering = saveState;

			UpdateImage();
		}

		private void InitializeViewModel()
		{
			label1.Text = "iteration: --";

			SetComboBox1Items();
			UpdateComboBox2Items();
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

		private string Float2(float value)
		{
			return string.Format(CultureInfo.InvariantCulture, "iteration: {0:F2} ms", value);
		}

		private void Timer_Tick(object sender, EventArgs e)
		{
			if(_isRendering)
			{
				if(daseffect != null && daseffect.IsValid())
				{
					var time = daseffect.Iteration(FramesPerOperation);

					label1.Text = Float2(time);

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

		private void SetComboBox1Items()
		{
			comboBox1.Items.Add("CPU C#");
			comboBox1.Items.Add("GPU Cuda C");
			comboBox1.SelectedIndex = 0;
		}

		private void UpdateComboBox2Items()
		{
			comboBox2.Items.Clear();

			if(daseffect != null)
			{
				List<string> list = daseffect.GetColorInterpretatorsTitle();

				if(list != null && list.Count > 0)
				{
					foreach(var title in list)
					{
						comboBox2.Items.Add(title);
					}

					comboBox2.SelectedIndex = 0;
				}
			}
		}

		private void pictureBox1_Click(object sender, EventArgs e)
		{
			_isRendering = !_isRendering;
		}

		private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
		{
			int index = comboBox1.SelectedIndex;

			if(index > 1)
			{
				index = 1;
			}

			InitializePhysicalModel(index);
			UpdateComboBox2Items();
		}

		private void comboBox2_SelectedIndexChanged(object sender, EventArgs e)
		{
			daseffect.SetColorInterpretator(comboBox2.SelectedIndex);
			UpdateImage();
		}
	}
}
