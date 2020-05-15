namespace User_Interface
{
	partial class Form1
	{
		/// <summary>
		/// Обязательная переменная конструктора.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		/// Освободить все используемые ресурсы.
		/// </summary>
		/// <param name="disposing">истинно, если управляемый ресурс должен быть удален; иначе ложно.</param>
		protected override void Dispose(bool disposing)
		{
			if (disposing && (components != null))
			{
				components.Dispose();
			}
			base.Dispose(disposing);
		}

		#region Код, автоматически созданный конструктором форм Windows

		/// <summary>
		/// Требуемый метод для поддержки конструктора — не изменяйте 
		/// содержимое этого метода с помощью редактора кода.
		/// </summary>
		private void InitializeComponent()
		{
			this.pictureBox1 = new System.Windows.Forms.PictureBox();
			this.label1 = new System.Windows.Forms.Label();
			this.comboBox2 = new System.Windows.Forms.ComboBox();
			this.comboBox1 = new System.Windows.Forms.ComboBox();
			this.label2 = new System.Windows.Forms.Label();
			this.label3 = new System.Windows.Forms.Label();
			this.trackBar1 = new System.Windows.Forms.TrackBar();
			this.label4 = new System.Windows.Forms.Label();
			this.label5 = new System.Windows.Forms.Label();
			this.trackBar2 = new System.Windows.Forms.TrackBar();
			this.label6 = new System.Windows.Forms.Label();
			this.trackBar3 = new System.Windows.Forms.TrackBar();
			this.textBox1 = new System.Windows.Forms.TextBox();
			this.textBox2 = new System.Windows.Forms.TextBox();
			this.textBox3 = new System.Windows.Forms.TextBox();
			this.button3 = new System.Windows.Forms.Button();
			this.button4 = new System.Windows.Forms.Button();
			this.textBox4 = new System.Windows.Forms.TextBox();
			this.label7 = new System.Windows.Forms.Label();
			this.trackBar4 = new System.Windows.Forms.TrackBar();
			this.button1 = new System.Windows.Forms.Button();
			this.button2 = new System.Windows.Forms.Button();
			this.button5 = new System.Windows.Forms.Button();
			((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.trackBar1)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.trackBar2)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.trackBar3)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.trackBar4)).BeginInit();
			this.SuspendLayout();
			// 
			// pictureBox1
			// 
			this.pictureBox1.Location = new System.Drawing.Point(12, 12);
			this.pictureBox1.Name = "pictureBox1";
			this.pictureBox1.Size = new System.Drawing.Size(512, 512);
			this.pictureBox1.TabIndex = 0;
			this.pictureBox1.TabStop = false;
			this.pictureBox1.Click += new System.EventHandler(this.pictureBox1_Click);
			// 
			// label1
			// 
			this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
			this.label1.Location = new System.Drawing.Point(374, 540);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(150, 23);
			this.label1.TabIndex = 1;
			this.label1.Text = "Iteration: 0 ms";
			this.label1.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// comboBox2
			// 
			this.comboBox2.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F);
			this.comboBox2.FormattingEnabled = true;
			this.comboBox2.Location = new System.Drawing.Point(545, 96);
			this.comboBox2.Name = "comboBox2";
			this.comboBox2.Size = new System.Drawing.Size(144, 24);
			this.comboBox2.TabIndex = 2;
			this.comboBox2.SelectedIndexChanged += new System.EventHandler(this.comboBox2_SelectedIndexChanged);
			// 
			// comboBox1
			// 
			this.comboBox1.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
			this.comboBox1.FormattingEnabled = true;
			this.comboBox1.Location = new System.Drawing.Point(545, 34);
			this.comboBox1.Name = "comboBox1";
			this.comboBox1.Size = new System.Drawing.Size(144, 24);
			this.comboBox1.TabIndex = 1;
			this.comboBox1.SelectedIndexChanged += new System.EventHandler(this.comboBox1_SelectedIndexChanged);
			// 
			// label2
			// 
			this.label2.AutoSize = true;
			this.label2.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
			this.label2.Location = new System.Drawing.Point(548, 12);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(134, 16);
			this.label2.TabIndex = 4;
			this.label2.Text = "Implementation Type";
			this.label2.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// label3
			// 
			this.label3.AutoSize = true;
			this.label3.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
			this.label3.Location = new System.Drawing.Point(548, 75);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(114, 16);
			this.label3.TabIndex = 5;
			this.label3.Text = "Color Interpretator";
			this.label3.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// trackBar1
			// 
			this.trackBar1.Location = new System.Drawing.Point(537, 251);
			this.trackBar1.Maximum = 20;
			this.trackBar1.Name = "trackBar1";
			this.trackBar1.Size = new System.Drawing.Size(144, 45);
			this.trackBar1.TabIndex = 5;
			this.trackBar1.Value = 19;
			this.trackBar1.Scroll += new System.EventHandler(this.trackBar1_Scroll);
			// 
			// label4
			// 
			this.label4.AutoSize = true;
			this.label4.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
			this.label4.Location = new System.Drawing.Point(540, 225);
			this.label4.Name = "label4";
			this.label4.Size = new System.Drawing.Size(101, 16);
			this.label4.TabIndex = 7;
			this.label4.Text = "Corruption Rate";
			this.label4.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// label5
			// 
			this.label5.AutoSize = true;
			this.label5.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
			this.label5.Location = new System.Drawing.Point(547, 143);
			this.label5.Name = "label5";
			this.label5.Size = new System.Drawing.Size(80, 16);
			this.label5.TabIndex = 9;
			this.label5.Text = "Water Level";
			this.label5.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// trackBar2
			// 
			this.trackBar2.Location = new System.Drawing.Point(544, 169);
			this.trackBar2.Maximum = 20;
			this.trackBar2.Name = "trackBar2";
			this.trackBar2.Size = new System.Drawing.Size(144, 45);
			this.trackBar2.TabIndex = 3;
			this.trackBar2.Value = 10;
			this.trackBar2.Scroll += new System.EventHandler(this.trackBar2_Scroll);
			// 
			// label6
			// 
			this.label6.AutoSize = true;
			this.label6.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
			this.label6.Location = new System.Drawing.Point(547, 312);
			this.label6.Name = "label6";
			this.label6.Size = new System.Drawing.Size(88, 16);
			this.label6.TabIndex = 11;
			this.label6.Text = "PhaseSpeed";
			this.label6.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// trackBar3
			// 
			this.trackBar3.Location = new System.Drawing.Point(544, 338);
			this.trackBar3.Maximum = 20;
			this.trackBar3.Name = "trackBar3";
			this.trackBar3.Size = new System.Drawing.Size(144, 45);
			this.trackBar3.TabIndex = 7;
			this.trackBar3.Value = 20;
			this.trackBar3.Scroll += new System.EventHandler(this.trackBar3_Scroll);
			// 
			// textBox1
			// 
			this.textBox1.Location = new System.Drawing.Point(694, 169);
			this.textBox1.Name = "textBox1";
			this.textBox1.Size = new System.Drawing.Size(50, 20);
			this.textBox1.TabIndex = 4;
			// 
			// textBox2
			// 
			this.textBox2.Location = new System.Drawing.Point(694, 251);
			this.textBox2.Name = "textBox2";
			this.textBox2.Size = new System.Drawing.Size(50, 20);
			this.textBox2.TabIndex = 6;
			// 
			// textBox3
			// 
			this.textBox3.Location = new System.Drawing.Point(694, 338);
			this.textBox3.Name = "textBox3";
			this.textBox3.Size = new System.Drawing.Size(50, 20);
			this.textBox3.TabIndex = 8;
			// 
			// button3
			// 
			this.button3.Location = new System.Drawing.Point(654, 481);
			this.button3.Name = "button3";
			this.button3.Size = new System.Drawing.Size(90, 43);
			this.button3.TabIndex = 12;
			this.button3.Text = "Open Folder";
			this.button3.UseVisualStyleBackColor = true;
			this.button3.Click += new System.EventHandler(this.button3_Click);
			// 
			// button4
			// 
			this.button4.Location = new System.Drawing.Point(543, 481);
			this.button4.Name = "button4";
			this.button4.Size = new System.Drawing.Size(90, 43);
			this.button4.TabIndex = 11;
			this.button4.Text = "Save Image";
			this.button4.UseVisualStyleBackColor = true;
			this.button4.Click += new System.EventHandler(this.button4_Click);
			// 
			// textBox4
			// 
			this.textBox4.Location = new System.Drawing.Point(694, 419);
			this.textBox4.Name = "textBox4";
			this.textBox4.Size = new System.Drawing.Size(50, 20);
			this.textBox4.TabIndex = 10;
			// 
			// label7
			// 
			this.label7.AutoSize = true;
			this.label7.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
			this.label7.Location = new System.Drawing.Point(547, 393);
			this.label7.Name = "label7";
			this.label7.Size = new System.Drawing.Size(54, 16);
			this.label7.TabIndex = 20;
			this.label7.Text = "Frames";
			this.label7.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// trackBar4
			// 
			this.trackBar4.LargeChange = 1;
			this.trackBar4.Location = new System.Drawing.Point(544, 419);
			this.trackBar4.Maximum = 32;
			this.trackBar4.Name = "trackBar4";
			this.trackBar4.Size = new System.Drawing.Size(144, 45);
			this.trackBar4.TabIndex = 9;
			this.trackBar4.Value = 4;
			this.trackBar4.Scroll += new System.EventHandler(this.trackBar4_Scroll);
			// 
			// button1
			// 
			this.button1.Location = new System.Drawing.Point(12, 530);
			this.button1.Name = "button1";
			this.button1.Size = new System.Drawing.Size(90, 43);
			this.button1.TabIndex = 0;
			this.button1.Text = "Start";
			this.button1.UseVisualStyleBackColor = true;
			this.button1.Click += new System.EventHandler(this.button1_Click);
			// 
			// button2
			// 
			this.button2.Location = new System.Drawing.Point(108, 530);
			this.button2.Name = "button2";
			this.button2.Size = new System.Drawing.Size(90, 43);
			this.button2.TabIndex = 21;
			this.button2.Text = "Tick";
			this.button2.UseVisualStyleBackColor = true;
			this.button2.Click += new System.EventHandler(this.button2_Click);
			// 
			// button5
			// 
			this.button5.Location = new System.Drawing.Point(204, 530);
			this.button5.Name = "button5";
			this.button5.Size = new System.Drawing.Size(90, 43);
			this.button5.TabIndex = 22;
			this.button5.Text = "Shot 60";
			this.button5.UseVisualStyleBackColor = true;
			this.button5.Click += new System.EventHandler(this.button5_Click);
			// 
			// Form1
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size(799, 603);
			this.Controls.Add(this.button5);
			this.Controls.Add(this.button2);
			this.Controls.Add(this.button1);
			this.Controls.Add(this.textBox4);
			this.Controls.Add(this.label7);
			this.Controls.Add(this.trackBar4);
			this.Controls.Add(this.button3);
			this.Controls.Add(this.button4);
			this.Controls.Add(this.textBox3);
			this.Controls.Add(this.textBox2);
			this.Controls.Add(this.textBox1);
			this.Controls.Add(this.label6);
			this.Controls.Add(this.trackBar3);
			this.Controls.Add(this.label5);
			this.Controls.Add(this.trackBar2);
			this.Controls.Add(this.label4);
			this.Controls.Add(this.trackBar1);
			this.Controls.Add(this.label3);
			this.Controls.Add(this.label2);
			this.Controls.Add(this.comboBox1);
			this.Controls.Add(this.comboBox2);
			this.Controls.Add(this.label1);
			this.Controls.Add(this.pictureBox1);
			this.Name = "Form1";
			this.Text = "Form1";
			((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.trackBar1)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.trackBar2)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.trackBar3)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.trackBar4)).EndInit();
			this.ResumeLayout(false);
			this.PerformLayout();

		}

		#endregion

		private System.Windows.Forms.PictureBox pictureBox1;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.ComboBox comboBox2;
		private System.Windows.Forms.ComboBox comboBox1;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.TrackBar trackBar1;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.Label label5;
		private System.Windows.Forms.TrackBar trackBar2;
		private System.Windows.Forms.Label label6;
		private System.Windows.Forms.TrackBar trackBar3;
		private System.Windows.Forms.TextBox textBox1;
		private System.Windows.Forms.TextBox textBox2;
		private System.Windows.Forms.TextBox textBox3;
		private System.Windows.Forms.Button button3;
		private System.Windows.Forms.Button button4;
		private System.Windows.Forms.TextBox textBox4;
		private System.Windows.Forms.Label label7;
		private System.Windows.Forms.TrackBar trackBar4;
		private System.Windows.Forms.Button button1;
		private System.Windows.Forms.Button button2;
		private System.Windows.Forms.Button button5;
	}
}

