using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Windows.Forms;
using Simple_2D_Landscape.LandscapeEngine;

namespace Simple_2D_Landscape
{
	public partial class Form1 : Form
	{
        private Daseffect _daseffect;

        private Timer _timer;

        private void SetPicture(Bitmap input, bool Scale = true)
        {
            int Width = pictureBox1.Width;
            int Height = pictureBox1.Height;

            Bitmap image = input;

            if(Scale && (input.Width != Width || input.Height != Height))
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

        private void InitializePhysicalModel()
        {
            _daseffect = new Daseffect(256, 256);
            SetPicture(_daseffect.GetBitmap());

           //pictureBox1.Image = _daseffect.GetBitmap();
        }

        private void InitializeTimer()
        {
            _timer = new Timer();

            _timer.Interval = 40;
            _timer.Enabled = false;

            _timer.Tick += new EventHandler(CalcTimerProcessor);
        }

        private void InitializeViewModel()
        {
            trackBar1.ValueChanged += TrackBar1_ValueChanged;

            trackBar1.Value = (int)(Math.Round(trackBar1.Maximum*_daseffect.WaterLevel / Daseffect.MaxWaterLevel));

            //comboBox1.DataSource =  Enum.GetValues(typeof(ColorInterpretationType));

            //comboBox1.SelectedItem = _daseffect.CurrentColorInterpretator;

            comboBox1.SelectedValueChanged += ComboBox1_SelectedValueChanged;

            textBox1.Text = _daseffect.RandomSeed.ToString();
        }

        private void ComboBox1_SelectedValueChanged(object sender, EventArgs e)
        {
            //_daseffect.CurrentColorInterpretator = (ColorInterpretationType)comboBox1.SelectedItem;

            //SetPicture(_daseffect.GetBitmap());
        }

        private void TrackBar1_ValueChanged(object sender, EventArgs e)
        {
            _daseffect.WaterLevel = (float)trackBar1.Value/trackBar1.Maximum;
            
            label2.Text = Math.Round(100*_daseffect.WaterLevel) + "%";

            //SetPicture(_daseffect.GetBitmap());
        }

        public Form1()
		{
			InitializeComponent();
            InitializePhysicalModel();
            InitializeViewModel();
            InitializeTimer();
		}

        private void CalcTimerProcessor(Object myObject, EventArgs myEventArgs)
        {
            Stopwatch sw = new Stopwatch();

            sw.Start();
            _daseffect.Iteration();
            //pictureBox1.Image = _daseffect.GetBitmap();
            //_daseffect.IterationOptimazed();
            SetPicture(_daseffect.GetBitmap());
            sw.Stop();
            
            long calcTime = sw.ElapsedMilliseconds;
        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {
            _timer.Enabled = !_timer.Enabled;
        }

        private void button7_Click(object sender, EventArgs e)
        {
            //_daseffect.IterationOptimazed();
            SetPicture(_daseffect.GetBitmap());
        }

        private void button8_Click(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {
            if(pictureBox1.Image != null)
            {
                try
                {
                    if(!Directory.Exists("Pictures"))
                        Directory.CreateDirectory("Pictures");

                    pictureBox1.Image.Save("Pictures\\" + 
                                           DateTime.Now.ToString("dd") + "d" + 
                                           DateTime.Now.ToString("MM") + "m" + 
                                           DateTime.Now.ToString("yy") + "y and " + 
                                           
                                           DateTime.Now.ToString("hh") + "h" + 
                                           DateTime.Now.ToString("mm") + "m" + 
                                           DateTime.Now.ToString("ss") + "s.png");
                }
                catch
                {

                }
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            if(!Directory.Exists("Pictures"))
                        Directory.CreateDirectory("Pictures");

            Process.Start(Directory.GetCurrentDirectory() + "\\Pictures");
        }

        //private void openGLControl1_OpenGLDraw_1(object sender, RenderEventArgs args)
        //{
        //    // Create a Simple Sample:

        //    OpenGL gl = openGLControl1.OpenGL;

        //    // Clear Screen & Depth Buffer
        //    gl.Clear(OpenGL.GL_COLOR_BUFFER_BIT | OpenGL.GL_DEPTH_BUFFER_BIT);

        //    // Reset World Matrix
        //    gl.LoadIdentity();

        //    // Move Draw Pointer to -Z coords.
        //    gl.Translate(0.0f, 0.0f, -5.0f);

        //    // Begin Drawing
        //    gl.Begin(OpenGL.GL_TRIANGLES);

        //    // Use White Color
        //    gl.Color(1.0f, 1.0f, 1.0f);

        //    gl.Vertex(-1.0f, -1.0f);
        //    gl.Vertex(0.0f, 1.0f);
        //    gl.Vertex(1.0f, -1.0f);

        //    // Draw triangle with points(vertex) { { -1.0f, -1.0f }, { 0.0f, 1.0f }, { 1.0f, -1.0f } }

        //    // Stop Drawing
        //    gl.End();
        //}
    }
}
