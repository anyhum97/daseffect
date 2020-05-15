using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Globalization;
using System.IO;
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
			UpdateItems();
		}

		private void InitializeViewModel()
		{
			label1.Text = "iteration:";

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

		private void UpdateItems()
		{
			if(daseffect != null && daseffect.IsValid())
			{
				textBox1.Text = Float2(daseffect.WaterLevel);
				textBox2.Text = Float2(daseffect.CorruptionRate);
				textBox3.Text = Float2(daseffect.PhaseSpeed);
				textBox4.Text = FramesPerOperation.ToString();

				trackBar4.Value = (int)((float)FramesPerOperation/MaxFramesPerOperation*trackBar4.Maximum+0.5f);
			}
		}

		private string Float2<Type>(Type value) where Type: struct
		{
			return string.Format(CultureInfo.InvariantCulture, "{0:F2}", value);
		}

		private void Timer_Tick(object sender, EventArgs e)
		{
			if(_isRendering)
			{
				if(daseffect != null && daseffect.IsValid())
				{
					var time = daseffect.Iteration(FramesPerOperation);

					label1.Text = "iteration: " + Float2(time) + "ms";

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

		private void Tick()
		{
			if(daseffect != null)
			{
				var time = daseffect.Iteration(1);
				label1.Text = "iteration: " + Float2(time) + "ms";
				UpdateImage();
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

		private void trackBar2_Scroll(object sender, EventArgs e)
		{
			float factor = (float)trackBar2.Value / trackBar2.Maximum;
			daseffect.WaterLevel = factor;
			trackBar2.Value = (int)(daseffect.WaterLevel * trackBar2.Maximum+0.5f);
			textBox1.Text = Float2(daseffect.WaterLevel);
			
			if(!_isRendering)
			{
				UpdateImage();
			}
		}

		private void trackBar1_Scroll(object sender, EventArgs e)
		{
			float factor = (float)trackBar1.Value / trackBar1.Maximum;
			daseffect.CorruptionRate = factor;
			trackBar1.Value = (int)(daseffect.CorruptionRate * trackBar2.Maximum+0.5f);
			textBox2.Text = Float2(daseffect.CorruptionRate);
		}

		private void trackBar3_Scroll(object sender, EventArgs e)
		{
			float factor = Daseffect.MaxPhaseSpeed * (float)trackBar3.Value / trackBar3.Maximum;
			daseffect.PhaseSpeed = factor;
			trackBar3.Value = (int)(daseffect.PhaseSpeed * trackBar2.Maximum / Daseffect.MaxPhaseSpeed + 0.5f);
			textBox3.Text = Float2(daseffect.PhaseSpeed);
		}

		private void trackBar4_Scroll(object sender, EventArgs e)
		{
			FramesPerOperation = trackBar4.Value*4;
			textBox4.Text = FramesPerOperation.ToString();
		}

		private void button4_Click(object sender, EventArgs e)
		{
			if(daseffect != null && daseffect.IsValid())
			{
				string directoryPath = daseffect.RandomSeed.ToString();

				try
				{
					if(!Directory.Exists(directoryPath))
					{
						Directory.CreateDirectory(directoryPath);
					}

					string filePath = daseffect.IterationCount.ToString();

					if(File.Exists(filePath))
					{
						File.Delete(filePath);
					}

					CurrentBitmap.Save(directoryPath + "\\" + filePath + ".png");

					_lastDirectoryPath = directoryPath;
				}
				catch
				{
					MessageBox.Show("Cannot Save the Image");
				}
			}
		}

		private void button3_Click(object sender, EventArgs e)
		{
			System.Diagnostics.Process.Start("explorer", _lastDirectoryPath);
		}

		private void button1_Click(object sender, EventArgs e)
		{
			_isRendering = !_isRendering;
		}

		private void button2_Click(object sender, EventArgs e)
		{
			Tick();
		}
	}
}
