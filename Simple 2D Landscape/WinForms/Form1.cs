using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using SharpGL;
using Simple_2D_Landscape.LandscapeEngine;

namespace Simple_2D_Landscape
{
	public partial class Form1 : Form
	{
        private static daseffect _test;
        private Timer _timer;

        private void SetPicture(Bitmap input)
        {
            int Width = pictureBox1.Width;
            int Height = pictureBox1.Height;

            Bitmap image = input;

            if(input.Width != Width || input.Height != Height)
            {
                image = new Bitmap(Width, Height);

                using(Graphics gr = Graphics.FromImage(image))
                {
                    gr.SmoothingMode = SmoothingMode.HighSpeed;
                    gr.CompositingQuality = CompositingQuality.HighSpeed;
                    gr.InterpolationMode = InterpolationMode.NearestNeighbor;
                    gr.DrawImage(input, new Rectangle(0, 0, Width, Height));
                }
            }

            pictureBox1.Image = image;
        }

		public Form1()
		{
			InitializeComponent();

            _test = new daseffect(128, 128);
            SetPicture(_test.GetBitmap());

            _timer = new Timer();
            _timer.Interval = 10;
            _timer.Enabled = true;

            _timer.Tick += new EventHandler(TimerEventProcessor);
		}

        private void TimerEventProcessor(Object myObject, EventArgs myEventArgs)
        {
            _test.Iteration();
            SetPicture(_test.GetBitmap());
        }

        private void openGLControl1_OpenGLDraw_1(object sender, RenderEventArgs args)
        {
            // Create a Simple Sample:

            OpenGL gl = openGLControl1.OpenGL;
            
            // Clear Screen & Depth Buffer
            gl.Clear(OpenGL.GL_COLOR_BUFFER_BIT | OpenGL.GL_DEPTH_BUFFER_BIT);
            
            // Reset World Matrix
            gl.LoadIdentity();
            
            // Move Draw Pointer to -Z coords.
            gl.Translate(0.0f, 0.0f, -5.0f);
            
            // Begin Drawing
            gl.Begin(OpenGL.GL_TRIANGLES);
            
            // Use White Color
            gl.Color(1.0f, 1.0f, 1.0f);
            
            gl.Vertex(-1.0f, -1.0f);
            gl.Vertex(0.0f, 1.0f);
            gl.Vertex(1.0f, -1.0f);
            
            // Draw triangle with points(vertex) { { -1.0f, -1.0f }, { 0.0f, 1.0f }, { 1.0f, -1.0f } }

            // Stop Drawing
            gl.End();
        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {
            _timer.Enabled = !_timer.Enabled;
        }
    }
}
