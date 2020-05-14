using SharpGL;
using SharpGL.SceneGraph;
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

namespace User_Interface
{
	public partial class MainWindow : Window
	{
		private uint[] _texture = new uint[2];

		private Timer _timer = new Timer();

		private bool _reloadPicture = true;

        public Daseffect daseffect { get; private set; }

		public Bitmap CurrentBitmap { get; private set; }

		public MainWindow()
		{
			InitializeComponent();

			daseffect = new Daseffect(256, 256);

			daseffect.Set(1, 128, 128, 1.0f);

			UpdateBitmap();

			_timer.Interval = 500;
			_timer.Elapsed += _timer_Elapsed;
			_timer.Start();
		}

		private void _timer_Elapsed(object sender, ElapsedEventArgs e)
		{
			daseffect.Iteration();

			UpdateBitmap();
			_reloadPicture = true;
		}

		private void UpdateBitmap()
		{
			CurrentBitmap = daseffect.GetBitmap();
		}

        private uint LoadTexture(OpenGL openGL, Bitmap bitmap)
        {
            uint[] textur = new uint[1];

            Rectangle rect = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
            
            var bitmapdata = bitmap.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            
            openGL.GenTextures(1, textur);

            openGL.BindTexture(OpenGL.GL_TEXTURE_2D, textur[0]);

            openGL.Build2DMipmaps(OpenGL.GL_TEXTURE_2D, (int)OpenGL.GL_RGBA, bitmap.Width, bitmap.Height, OpenGL.GL_BGR_EXT, OpenGL.GL_UNSIGNED_BYTE, bitmapdata.Scan0);

			openGL.TexParameter(OpenGL.GL_TEXTURE_2D, OpenGL.GL_TEXTURE_MAG_FILTER, OpenGL.GL_NEAREST);
            openGL.TexParameter(OpenGL.GL_TEXTURE_2D, OpenGL.GL_TEXTURE_MIN_FILTER, OpenGL.GL_LINEAR_MIPMAP_NEAREST);
            
            uint tex = textur[0];
            
            bitmap.UnlockBits(bitmapdata);
            bitmap.Dispose();
            
			return tex;
        }

		private void OpenGLControl_OpenGLInitialized(object sender, OpenGLEventArgs args)
		{
			OpenGL openGL = OpenGLControl.OpenGL;

			openGL.Color(1.0f, 1.0f, 1.0f);
			
            openGL.Enable(OpenGL.GL_TEXTURE_2D);
			
            _texture[0] = LoadTexture(openGL, CurrentBitmap);
		}

		private void OpenGLControl_OpenGLDraw(object sender, OpenGLEventArgs args)
		{
			// OpenGL openGL = OpenGLControl.OpenGL;

			OpenGL openGL = args.OpenGL;

			if(_reloadPicture)
			{
				//LoadTexture(openGL, CurrentBitmap);
				_reloadPicture = false;
			}

			openGL.Clear(OpenGL.GL_COLOR_BUFFER_BIT | OpenGL.GL_DEPTH_BUFFER_BIT);
            
            openGL.ClearColor(1.0f, 1.0f, 1.0f, 0.0f);

			openGL.Begin(OpenGL.GL_QUADS);

			//openGL.Color(1.0f, 1.0f, 1.0f);
			//
			//openGL.TexCoord(1.0f, 0.0f); openGL.Vertex(-1.0f, -1.0f, -1.0f);
			//openGL.TexCoord(1.0f, 1.0f); openGL.Vertex(-1.0f, 1.0f, -1.0f);
			//openGL.TexCoord(0.0f, 1.0f); openGL.Vertex(1.0f, 1.0f, -1.0f);
			//openGL.TexCoord(0.0f, 0.0f); openGL.Vertex(1.0f, -1.0f, -1.0f);
            
			openGL.End();
			//openGL.Flush();
		}

		private void OpenGLControl_Resized(object sender, OpenGLEventArgs args)
		{
			OpenGL openGL = args.OpenGL;
		}
	}
}
