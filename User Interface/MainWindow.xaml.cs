using OpenTK.Graphics.ES20;
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
using Utils;

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

			Timer timer = new Timer();

			timer.Interval = 100;
			timer.Elapsed += Timer_Elapsed;
			timer.Start();
		}

         private bool canDraw = false;
        private int program;
        private int nVertices;

		private void Timer_Elapsed(object sender, ElapsedEventArgs e)
		{
			daseffect.Iteration();
			//MainImage.Source = ImageToByte(daseffect.GetBitmap());
		}

		private void WindowsFormsHost_Initialized(object sender, EventArgs e)
        {
            //renderCanvas.MakeCurrent();
        }

        private void renderCanvas_Load(object sender, EventArgs e)
        {
            // Load shaders from files

            string vShaderSource = null;
            string fShaderSource = null;

            ShaderLoader.LoadShader("Shaders/VertexShader.glsl", out vShaderSource);
            ShaderLoader.LoadShader("Shaders/FragmentShader.glsl", out fShaderSource);

            if (vShaderSource == null || fShaderSource == null)
            {
                return;
            }

            // Initialize the shaders
            if (!ShaderLoader.InitShaders(vShaderSource, fShaderSource, out program))
            {
                return;
            }

            // Write the positions of vertices to a vertex shader
            nVertices = InitVertexBuffers();
            if (nVertices <= 0)
            {
                return;
            }

            // Specify the color for clearing
            GL.ClearColor(System.Drawing.Color.DarkSlateBlue);

            canDraw = true;
        }

        private int InitVertexBuffers()
        {
            float[] vertices = new float[] { 0f, 0.5f, -0.5f, -0.5f, 0.5f, -0.5f };

            // Create a buffer object
            int vertexBuffer;
            GL.GenBuffers(1, out vertexBuffer);

            // Bind the buffer object to target
            GL.BindBuffer(BufferTarget.ArrayBuffer, vertexBuffer);

            // Write data into the buffer object
            GL.BufferData(BufferTarget.ArrayBuffer, vertices.Length * sizeof(float), vertices, BufferUsageHint.StaticDraw);

            // Get the storage location of a_Position
            int a_Position = GL.GetAttribLocation(program, "a_Position");
            if (a_Position < 0)
            {
                return -1;
            }

            // Assign the buffer object to a_Position variable
            GL.VertexAttribPointer(a_Position, 2, VertexAttribPointerType.Float, false, 0, 0);

            // Enable the assignment to a_Position variable
            GL.EnableVertexAttribArray(a_Position);

            return vertices.Length / 2;
        }

        private void renderCanvas_Paint(object sender, EventArgs e)
        {
            GL.Viewport(0, 0, renderCanvas.Width, renderCanvas.Height);

            // Clear the render canvas with the current color
            GL.Clear(ClearBufferMask.ColorBufferBit);

            if (canDraw)
            {
                // Draw a triangle
                GL.DrawArrays(PrimitiveType.Triangles, 0, nVertices);
            }

            GL.Flush();
            renderCanvas.SwapBuffers();
        }
	}
}
