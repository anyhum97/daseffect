using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
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
        private daseffect test;

		public Form1()
		{
			InitializeComponent();

            test = new daseffect(128, 256);
            pictureBox1.Image = test.GetBitmap();
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
            test.Iteration();
            pictureBox1.Image = test.GetBitmap();
        }
    }
}
