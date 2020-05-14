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
		internal static ImageSource ImageToByte(Bitmap bitmap)
		{
			return System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(
			bitmap.GetHbitmap(),
			IntPtr.Zero,
			Int32Rect.Empty,
			BitmapSizeOptions.FromEmptyOptions());
		}

		private Daseffect daseffect;

		public MainWindow()
		{
			InitializeComponent();

			daseffect = new Daseffect(256, 256);

			daseffect.Set(1, 128, 128, 1.0f);

			//MainImage.Source = ImageToByte(daseffect.GetBitmap());

			var timer = new System.Timers.Timer();

			timer.Interval = 100;
			timer.Elapsed += Timer_Elapsed;
			timer.Start();
		}

		private void Timer_Elapsed(object sender, ElapsedEventArgs e)
		{
			daseffect.Iteration();
			//MainImage.Source = ImageToByte(daseffect.GetBitmap());
		}


		private bool drawred;

		private System.Timers.Timer fpstimer;
		private DispatcherTimer rendertimer;
		private Thread renderthread;
		private int fps;

		private delegate void fpsdelegate();
		private void showfps()
		{
			this.Title = "FPS: " + fps; fps = 0;
		}

		private void Window_Loaded_1(object sender, RoutedEventArgs e)
		{
			fpstimer = new System.Timers.Timer(1000);
			fpstimer.Elapsed += (sender1, args) =>
			{
				Dispatcher.BeginInvoke(DispatcherPriority.Render, new fpsdelegate(showfps));
			};
			fpstimer.Start();

			//// !! uncomment for regular FPS renderloop !!
			//rendertimer = new DispatcherTimer();
			//rendertimer.Interval = TimeSpan.FromMilliseconds(15); /* ~60Hz LCD on my PC */
			//rendertimer.Tick += (o, args) => Render();
			//rendertimer.Start();

			// !! comment for maximum FPS renderloop !!
			renderthread = new Thread(() =>
			{
				while (true)
					Render();
			});
			renderthread.Start();
		}

		private void Render()
		{
			// do lock to avoid resize/repaint race in control
			// where are BMP and GFX recreates
			// better practice is Monitor.TryEnter() pattern, but here we do it simpler
			lock (razorPainterWPFCtl1.RazorLock)
			{
				razorPainterWPFCtl1.RazorGFX.Clear((drawred = !drawred) ? System.Drawing.Color.Red : System.Drawing.Color.Blue);
				razorPainterWPFCtl1.RazorGFX.DrawString("habrahabr.ru", System.Drawing.SystemFonts.DefaultFont, System.Drawing.Brushes.Azure, 10, 10);
				razorPainterWPFCtl1.RazorPaint();
			}
			fps++;
		}

		private void Window_Closing_1(object sender, System.ComponentModel.CancelEventArgs e)
		{
			renderthread.Abort();
			//rendertimer.Stop();
			fpstimer.Stop();
		}
	}
}
