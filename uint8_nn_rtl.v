// Uint8 量化神经网络 RTL实现参考
// 两层全连接网络: 48 -> 16 -> 3

module uint8_nn_top (
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire [7:0] pixel_data [0:47],  // 48个uint8输入
    output reg valid_out,
    output reg [1:0] class_out           // 分类结果 0/1/2
);

// 内部信号
wire [7:0] hidden [0:15];      // Layer 1 输出 (16个uint8)
wire layer1_done;
wire layer2_done;

// Layer 1: 48 -> 16
fc_layer_48x16 layer1 (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(valid_in),
    .input_data(pixel_data),
    .output_data(hidden),
    .valid_out(layer1_done)
);

// Layer 2: 16 -> 3
fc_layer_16x3 layer2 (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(layer1_done),
    .input_data(hidden),
    .class_out(class_out),
    .valid_out(layer2_done)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        valid_out <= 1'b0;
    else
        valid_out <= layer2_done;
end

endmodule


// ============================================
// 全连接层: 48输入 -> 16输出
// ============================================
module fc_layer_48x16 (
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire [7:0] input_data [0:47],
    output reg [7:0] output_data [0:15],
    output reg valid_out
);

// 权重存储 (ROM) - 从训练得到
reg [7:0] weights [0:15][0:47];  // 16×48 权重矩阵
reg [7:0] bias [0:15];           // 16个偏置

// 状态机
localparam IDLE = 2'b00;
localparam COMPUTE = 2'b01;
localparam DONE = 2'b10;

reg [1:0] state;
reg [4:0] neuron_idx;  // 0-15 神经元索引

// MAC (乘累加) 结果
reg [23:0] mac_result;  // 8×8×48 最大需要约20bit，留余量
reg [5:0] mac_cnt;      // 0-47 计数器

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        valid_out <= 1'b0;
        neuron_idx <= 0;
        mac_cnt <= 0;
    end else begin
        case (state)
            IDLE: begin
                valid_out <= 1'b0;
                if (valid_in) begin
                    state <= COMPUTE;
                    neuron_idx <= 0;
                    mac_cnt <= 0;
                    mac_result <= 0;
                end
            end
            
            COMPUTE: begin
                // MAC: result += input[i] × weight[neuron_idx][i]
                mac_result <= mac_result + 
                             (input_data[mac_cnt] * weights[neuron_idx][mac_cnt]);
                
                if (mac_cnt == 47) begin
                    // 一个神经元计算完成
                    // 加bias，除以256缩放，ReLU
                    reg [15:0] temp;
                    temp = (mac_result >> 8) + bias[neuron_idx];
                    
                    // ReLU + 饱和
                    if (temp[15]) // 负数
                        output_data[neuron_idx] <= 8'h00;
                    else if (temp > 255)
                        output_data[neuron_idx] <= 8'hFF;
                    else
                        output_data[neuron_idx] <= temp[7:0];
                    
                    mac_cnt <= 0;
                    mac_result <= 0;
                    
                    if (neuron_idx == 15) begin
                        state <= DONE;
                    end else begin
                        neuron_idx <= neuron_idx + 1;
                    end
                end else begin
                    mac_cnt <= mac_cnt + 1;
                end
            end
            
            DONE: begin
                valid_out <= 1'b1;
                state <= IDLE;
            end
            
            default: state <= IDLE;
        endcase
    end
end

// 权重初始化 (从文件或参数加载)
initial begin
    // 这里应该从训练导出的权重文件加载
    // $readmemh("fc1_weights.hex", weights);
    // $readmemh("fc1_bias.hex", bias);
end

endmodule


// ============================================
// 全连接层: 16输入 -> 3输出 (带argmax)
// ============================================
module fc_layer_16x3 (
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire [7:0] input_data [0:15],
    output reg [1:0] class_out,
    output reg valid_out
);

reg [7:0] weights [0:2][0:15];  // 3×16 权重矩阵
reg [7:0] bias [0:2];           // 3个偏置

reg [1:0] state;
reg [1:0] neuron_idx;
reg [15:0] mac_result;
reg [3:0] mac_cnt;

// 输出logits
reg [15:0] logits [0:2];

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        valid_out <= 1'b0;
        neuron_idx <= 0;
        mac_cnt <= 0;
    end else begin
        case (state)
            IDLE: begin
                valid_out <= 1'b0;
                if (valid_in) begin
                    state <= COMPUTE;
                    neuron_idx <= 0;
                    mac_cnt <= 0;
                    mac_result <= 0;
                end
            end
            
            COMPUTE: begin
                mac_result <= mac_result + 
                             (input_data[mac_cnt] * weights[neuron_idx][mac_cnt]);
                
                if (mac_cnt == 15) begin
                    logits[neuron_idx] <= (mac_result >> 8) + bias[neuron_idx];
                    
                    mac_cnt <= 0;
                    mac_result <= 0;
                    
                    if (neuron_idx == 2) begin
                        state <= DONE;
                    end else begin
                        neuron_idx <= neuron_idx + 1;
                    end
                end else begin
                    mac_cnt <= mac_cnt + 1;
                end
            end
            
            DONE: begin
                // Argmax
                if (logits[0] >= logits[1] && logits[0] >= logits[2])
                    class_out <= 2'b00;
                else if (logits[1] >= logits[2])
                    class_out <= 2'b01;
                else
                    class_out <= 2'b10;
                
                valid_out <= 1'b1;
                state <= IDLE;
            end
            
            default: state <= IDLE;
        endcase
    end
end

endmodule


// ============================================
// 优化版本：使用并行MAC单元
// ============================================
module fc_layer_48x16_parallel (
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire [7:0] input_data [0:47],
    output reg [7:0] output_data [0:15],
    output reg valid_out
);

// 16个并行MAC单元
genvar i;
generate
    for (i = 0; i < 16; i = i + 1) begin : mac_units
        wire [23:0] mac_out;
        
        mac_48_inputs mac_inst (
            .clk(clk),
            .rst_n(rst_n),
            .valid_in(valid_in),
            .input_data(input_data),
            .weights(weights[i]),
            .bias(bias[i]),
            .result(mac_out),
            .valid_out(/* 所有MAC同时完成 */)
        );
        
        // ReLU
        always @(*) begin
            if (mac_out[23]) // 负数
                output_data[i] = 8'h00;
            else if (mac_out > 255)
                output_data[i] = 8'hFF;
            else
                output_data[i] = mac_out[7:0];
        end
    end
endgenerate

// 延迟valid信号 (1个周期完成计算)
always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        valid_out <= 1'b0;
    else
        valid_out <= valid_in;
end

endmodule


// ============================================
// 单个MAC单元：48个输入的点积
// ============================================
module mac_48_inputs (
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire [7:0] input_data [0:47],
    input wire [7:0] weights [0:47],
    input wire [7:0] bias,
    output reg [23:0] result,
    output reg valid_out
);

integer i;
reg [23:0] sum;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        result <= 0;
        valid_out <= 1'b0;
    end else if (valid_in) begin
        sum = 0;
        for (i = 0; i < 48; i = i + 1) begin
            sum = sum + (input_data[i] * weights[i]);
        end
        result <= (sum >> 8) + bias;
        valid_out <= 1'b1;
    end else begin
        valid_out <= 1'b0;
    end
end

endmodule
