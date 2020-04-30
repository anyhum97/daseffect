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

namespace Simple_2D_Landscape
{
	public partial class Form1 : Form
	{
		public Form1()
		{
			InitializeComponent();
		}

        private void openGLControl1_OpenGLDraw_1(object sender, RenderEventArgs args)
        {
			// Создаем экземпляр
            OpenGL gl = this.openGLControl1.OpenGL;
            
            // Очистка экрана и буфера глубин
            gl.Clear(OpenGL.GL_COLOR_BUFFER_BIT | OpenGL.GL_DEPTH_BUFFER_BIT);
            
            // Сбрасываем модельно-видовую матрицу
            gl.LoadIdentity();
            
            // Двигаем перо вглубь экрана
            gl.Translate(0.0f, 0.0f, -5.0f);
            
            gl.Begin(OpenGL.GL_TRIANGLES);
            
            // Указываем цвет вершин
            gl.Color(1f, 1f, 1f);
            
            gl.Vertex(-1f, -1f);
            gl.Vertex(0f, 1f);
            gl.Vertex(1f, -1f);
            
            // Завершаем работу
            gl.End();
        }
    }
}
