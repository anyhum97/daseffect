using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
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

			MainImage.Source = ImageToByte(daseffect.GetBitmap());

			Timer timer = new Timer();

			timer.Interval = 100;
			timer.Elapsed += Timer_Elapsed;
			timer.Start();
		}

		private void Timer_Elapsed(object sender, ElapsedEventArgs e)
		{
			daseffect.Iteration();
			//MainImage.Source = ImageToByte(daseffect.GetBitmap());
		}
	}
}
