using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Timers;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Threading;

namespace User_Interface
{
	/// <summary>
	/// Логика взаимодействия для MainWindow.xaml
	/// </summary>
	public partial class MainWindow : Window
	{
		private CudaAdaptor daseffect;

		private System.Timers.Timer timer = new System.Timers.Timer();

		public MainWindow()
		{
			InitializeComponent();

			daseffect = new CudaAdaptor(512, 512);

			UpdateImage();

			timer.Interval = 100;
			timer.Elapsed += Timer_Elapsed;
			timer.Start();
		}

		private void Timer_Elapsed(object sender, ElapsedEventArgs e)
		{
			float time = daseffect.Iteration();
			UpdateImage();
			Label1.Content = time.ToString();

			timer.Interval = (int)(daseffect.IterationTime + daseffect.FrameTime + 1.0f);
		}

		private void UpdateImage()
		{
			razorPainterWPFCtl1.RazorBMP = daseffect.GetBitmap();
			razorPainterWPFCtl1.RazorPaint();
		}
	}
}
