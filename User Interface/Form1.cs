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
		#region Fields

		private DaseffectBase daseffect;

		private Timer _timer;

		private string _lastDirectoryPath = null;

		private int _garbageCollectorCount = default;
		
		private int GarbageCollectorCount
		{
			get => _garbageCollectorCount;
			
			set
			{
				_garbageCollectorCount = value;
				
				if(_garbageCollectorCount >= 16)
				{
					GC.Collect();
					_garbageCollectorCount = 0;
				}
			}
		}

		public Bitmap CurrentBitmap { get; set; }

		#endregion

		#region NotifyPropertyChanged

		private void NotifyPropertyChanged(EventHandler eventHandler)
		{
			if(eventHandler != null)
			{
				eventHandler.Invoke(this, EventArgs.Empty);
			}
		}

		#region IsRendering

		private bool _isRendering = false;

		public event EventHandler NotifyIsRenderingChanged;

		public bool IsRendering
		{
			get => _isRendering;

			private set
			{

				if(_isRendering != value)
				{
					_isRendering = value;
					NotifyPropertyChanged(NotifyIsRenderingChanged);
				}
			}
		}

		#endregion

		#region FramesPerOperation

		public const int DefaultFramesPerOperation = 1;

		public const int MinFramesPerOperation = 1;
		public const int MaxFramesPerOperation = 128;

		private int _framesPerOperation = DefaultFramesPerOperation;

		public event EventHandler NotifyFramesPerOperationChanged;

		public int FramesPerOperation
		{
			get => _framesPerOperation;
			
			set
			{
				int newValue = value;

				if(newValue < MinFramesPerOperation)
				{
					newValue = MinFramesPerOperation;
				}

				if(newValue > MaxFramesPerOperation)
				{
					newValue = MaxFramesPerOperation;
				}

				if(_framesPerOperation != newValue)
				{
					_framesPerOperation = newValue;
					NotifyPropertyChanged(NotifyFramesPerOperationChanged);
				}
			}
		}

		#endregion

		#region WaterLevel

		private float _waterLevel = DaseffectBase.DefaultWaterLevel;

		public event EventHandler NotifyWaterLevelChanged;

		public float WaterLevel
		{
			get => _waterLevel;

			set
			{
				if(daseffect != null)
				{
					daseffect.WaterLevel = value;

					if(_waterLevel != daseffect.WaterLevel)
					{
						_waterLevel = daseffect.WaterLevel;
						NotifyPropertyChanged(NotifyWaterLevelChanged);
					}
				}
				else
				{
					_waterLevel = value;
				}
			}
		}

		#endregion

		#region CorruptionRate

		private double _corruptionRate = DaseffectBase.DefaultCorruptionRate;

		public event EventHandler NotifyCorruptionRateChanged;

		public double CorruptionRate
		{
			get => _corruptionRate;

			set
			{
				if(daseffect != null)
				{
					daseffect.CorruptionRate = value;

					if(_corruptionRate != daseffect.CorruptionRate)
					{
						_corruptionRate = daseffect.CorruptionRate;
						NotifyPropertyChanged(NotifyCorruptionRateChanged);
					}
				}
				else
				{
					_corruptionRate = value;
				}
			}
		}

		#endregion

		#region PhaseSpeed

		private float _phaseSpeed = DaseffectBase.DefaultPhaseSpeed;

		public event EventHandler NotifyPhaseSpeedChanged;

		public float PhaseSpeed
		{
			get => _phaseSpeed;

			set
			{
				if(daseffect != null)
				{
					daseffect.PhaseSpeed = value;

					if(_phaseSpeed != daseffect.PhaseSpeed)
					{
						_phaseSpeed = daseffect.PhaseSpeed;
						NotifyPropertyChanged(NotifyPhaseSpeedChanged);
					}
				}
				else
				{
					_phaseSpeed = value;
				}
			}
		}

		#endregion

		#endregion

		public Form1()
		{
			InitializeComponent();
			InitializeViewModel();
			StartTimer();

			KeyPreview = true;
			KeyDown += Form1_KeyDown;
		}

		private void InitializeViewModel()
		{
			NotifyIsRenderingChanged += Form1_NotifyIsRenderingChanged;
			
			NotifyWaterLevelChanged += Form1_NotifyWaterLevelChanged;		
			NotifyCorruptionRateChanged += Form1_NotifyCorruptionRateChanged;
			NotifyPhaseSpeedChanged += Form1_NotifyPhaseSpeedChanged;

			NotifyFramesPerOperationChanged += Form1_NotifyFramesPerOperationChanged;

			UpdateComboBox1Items();
			UpdateComboBox2Items();
		}

		private void Form1_NotifyIsRenderingChanged(object sender, EventArgs e)
		{
			button1.Text = IsRendering ? "Stop" : "Start";
		}

		private void Form1_NotifyWaterLevelChanged(object sender, EventArgs e)
		{
			textBox1.Text = Float2(daseffect.WaterLevel);
			
			if(!IsRendering)
			{
				UpdateImage();
			}
		}

		private void Form1_NotifyCorruptionRateChanged(object sender, EventArgs e)
		{
			textBox2.Text = Float2(CorruptionRate);
		}
		
		private void Form1_NotifyPhaseSpeedChanged(object sender, EventArgs e)
		{
			textBox3.Text = Float2(PhaseSpeed);
		}

		private void Form1_NotifyFramesPerOperationChanged(object sender, EventArgs e)
		{
			textBox4.Text = FramesPerOperation.ToString();

			trackBar4.Value = (int)Math.Round((double)FramesPerOperation/MaxFramesPerOperation*trackBar4.Maximum);
		}
		
		private void UpdateComboBox1Items()
		{
			comboBox1.Items.Clear();

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

					if(comboBox2.Items.Count > 3)
					{
						comboBox2.SelectedIndex = 3;
					}
					else
					{
						comboBox2.SelectedIndex = 0;
					}
				}
			}
		}

		private void StartTimer(int interval = 100)
		{
			_timer = new Timer();

			_timer.Interval = interval;
			_timer.Tick += Timer_Tick;

			_timer.Start();
		}

		private void Timer_Tick(object sender, EventArgs e)
		{
			if(IsRendering)
			{
				if(daseffect != null)
				{
					var time = daseffect.Iteration(FramesPerOperation);

					label1.Text = "iteration: " + Float2(time) + "ms";

					UpdateImage();
				}
			}
		}

		private void UpdateImage()
		{
			CurrentBitmap = null;

			if(daseffect != null)
			{
				CurrentBitmap = daseffect.GetBitmap();
				
				if(CurrentBitmap != null)
				{
					++GarbageCollectorCount;
				}
			}
			
			pictureBox1.Image = CurrentBitmap;
		}

		private void UpdateItems()
		{
			if(daseffect != null && daseffect.IsValid())
			{
				NotifyPropertyChanged(NotifyWaterLevelChanged);
				NotifyPropertyChanged(NotifyCorruptionRateChanged);
				NotifyPropertyChanged(NotifyPhaseSpeedChanged);
				NotifyPropertyChanged(NotifyFramesPerOperationChanged);
			}
		}

		private void InitializePhysicalModel(int modelType = 0)
		{
			bool saveState = IsRendering;

			IsRendering = false;

			if(modelType == 0)
			{
				daseffect = new Daseffect(512, 512);
			}
			else
			{
				daseffect = new CudaAdapter(512, 512);
			}

			daseffect.AddNoise(0.001f, 0.1f, 0.005f);

			IsRendering = saveState;

			UpdateImage();
			UpdateItems();
		}

		private void Form1_KeyDown(object sender, KeyEventArgs e)
		{
			if(e.KeyCode == Keys.D1)
			{
				Tick();
			}

			if(e.KeyCode == Keys.D2)
			{
				SaveImage();
			}
		}

		private void pictureBox1_Click(object sender, EventArgs e)
		{
			IsRendering = !IsRendering;
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

		private void trackBar1_Scroll(object sender, EventArgs e)
		{
			CorruptionRate = (float)trackBar1.Value / trackBar1.Maximum;

			trackBar1.Value = (int)Math.Round(CorruptionRate * trackBar2.Maximum);
		}

		private void trackBar2_Scroll(object sender, EventArgs e)
		{			
			WaterLevel = (float)trackBar2.Value / trackBar2.Maximum;

			trackBar2.Value = (int)Math.Round(WaterLevel * trackBar2.Maximum);
		}

		private void trackBar3_Scroll(object sender, EventArgs e)
		{
			PhaseSpeed = DaseffectBase.MaxPhaseSpeed * (float)trackBar3.Value / trackBar3.Maximum;

			trackBar3.Value = (int)Math.Round(PhaseSpeed * trackBar2.Maximum / Daseffect.MaxPhaseSpeed);
		}

		private void trackBar4_Scroll(object sender, EventArgs e)
		{
			FramesPerOperation = trackBar4.Value*4;
		}

		private void button1_Click(object sender, EventArgs e)
		{
			IsRendering = !IsRendering;
		}

		private void button2_Click(object sender, EventArgs e)
		{
			Tick();
		}

		private void button3_Click(object sender, EventArgs e)
		{
			System.Diagnostics.Process.Start("explorer", _lastDirectoryPath);
		}

		private void button4_Click(object sender, EventArgs e)
		{
			SaveImage();
		}

		private void button5_Click(object sender, EventArgs e)
		{
			for(int i=0; i<60; ++i)
			{
				Tick();
				SaveImage();
			}
		}

		private void button6_Click(object sender, EventArgs e)
		{
			int index = comboBox1.SelectedIndex;

			if(index > 1)
			{
				index = 1;
			}

			InitializePhysicalModel(index);
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

		private void SaveImage()
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

		private string Float2<Type>(Type value) where Type: struct
		{
			return string.Format(CultureInfo.InvariantCulture, "{0:F2}", value);
		}

	}
}
