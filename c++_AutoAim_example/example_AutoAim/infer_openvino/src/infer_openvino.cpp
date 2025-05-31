#include "infer_openvino.hpp"

cv::Mat formatToSquare(const cv::Mat &source)
{
	int col = source.cols;
	// std::cout << "col:" << col << std::endl;

	int row = source.rows;
	int _max = MAX(col, row);
	// std::cout << "_max:" << _max << std::endl;

	cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
	source.copyTo(result(cv::Rect(0, 0, col, row)));
	return result;
}

infer_vino::infer_vino(std::string model_xml, std::string device)
{

	// -------- Step 1. Initialize OpenVINO Runtime Core --------
	// 初始化OpenVINO Core并加载模型
	ov::Core core;
	auto model = core.read_model(model_xml);
	// _mode = mode;
	// -------- Step 2. Compile the Model --------
	// 固定输入形状为[1,3,640,640]
	// ov::PartialShape input_shape = model->input().get_partial_shape();
	// input_shape[0] = 1;   // 固定batch为1
	// input_shape[2] = 640; // 高度
	// input_shape[3] = 640; // 宽度
	// model->reshape({{1, 3, 640, 640}});
	compiled_model = core.compile_model(model, device);
	// -------- Step 3. Create an Inference Request --------
	request = compiled_model.create_infer_request();
}

std::vector<std::vector<float>> infer_vino::vector_pred(const std::vector<std::vector<float>> &input_data, int input_step, int input_dim)
{
	try
	{

		// -------- 数据验证 --------
		// 检查输入是否为空
		if (input_data.empty())
		{
			throw std::invalid_argument("Input data is empty");
		}

		// 检查维度是否匹配
		if (input_data.size() != input_step || input_data[0].size() != input_dim)
		{
			std::ostringstream error_msg;
			error_msg << "Input data must be a " << input_step << "x" << input_dim << " vector. "
					  << "Actual dimensions: " << input_data.size() << "x" << input_data[0].size();
			throw std::invalid_argument(error_msg.str()); // 使用 .str() 获取字符串
		}
		
		// -------- 创建输入张量 --------
		auto input_port = compiled_model.input();
		ov::Shape input_shape = input_port.get_shape();

		// 创建一个连续的一维数组来存储输入数据
		std::vector<float> contiguous_data;
		contiguous_data.reserve(input_data.size() * input_data[0].size());
		for (const auto &row : input_data)
		{
			contiguous_data.insert(contiguous_data.end(), row.begin(), row.end());
		}

		// 创建张量并拷贝数据
		ov::Tensor input_tensor(input_port.get_element_type(), input_shape);
		std::memcpy(input_tensor.data<float>(), contiguous_data.data(), contiguous_data.size() * sizeof(float));

		// -------- 设置输入并推理 --------
		request.set_input_tensor(input_tensor);
		request.infer();

		// -------- 处理输出 --------
		ov::Tensor output_tensor = request.get_output_tensor();
		const float *output_data = output_tensor.data<const float>();
		// std::cout << "output_tensor: " << output_tensor.get_shape() << std::endl; // output_tensor: [1,12,3]

		// 将输出转换为二维vector [12,3]
		// const size_t output_batch = output_tensor.get_shape()[0];
		const size_t output_rows = output_tensor.get_shape()[1];
		const size_t output_cols = output_tensor.get_shape()[2];
		std::vector<std::vector<float>> output_vector(output_rows, std::vector<float>(output_cols));

		for (size_t i = 0; i < output_rows; ++i)
		{
			for (size_t j = 0; j < output_cols; ++j)
			{
				output_vector[i][j] = output_data[i * output_cols + j];
			}
		}

		return output_vector;
	}
	catch (const std::exception &e)
	{
		std::cerr << "Inference failed: " << e.what() << std::endl;
		return {};
	}
}

std::vector<Detection> infer_vino::image_detect(cv::Mat &frame)
{

	// -------- Step 4.Read a picture file and do the preprocess --------
	// 训练onnx模型时输入的图像数据 batch=1,channel=3,height=640,width=640

	cv::Mat modelInput = formatToSquare(frame);

	cv::Mat blob_image;

	cv::dnn::blobFromImage(modelInput, blob_image, 1.0 / 255.0, cv::Size(320, 320), cv::Scalar(), true, false);

	// -------- Step 5. Feed the blob into the input node of the Model -------
	// Get input port for model with one input
	auto input_port = compiled_model.input();
	// Create tensor from external memory

	ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob_image.ptr(0));
	// Set input tensor for model with one input
	request.set_input_tensor(input_tensor);

	// -------- Step 6. Start inference -------
	request.infer();

	// -------- Step 7. Get the inference result --------
	ov::Tensor output = request.get_output_tensor();

	const float *prob = (float *)output.data();

	const ov::Shape outputDims = output.get_shape();
	// std::cout << "The shape of output tensor:" << outputDims << std::endl;

	size_t numRows = outputDims[1]; // cxcywh+classes+keypoints*2
	size_t numCols = outputDims[2]; // 8400=20*20+40*40+80*80

	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<std::vector<cv::Point2f>> objects_keypoints;
	cv::Mat det_output(numRows, numCols, CV_32F, (float *)prob);

	cv::transpose(det_output, det_output); // [cxcywh+classes+keypoints*2,8400]--->[8400,cxcywh+classes+keypoints*2]
	for (int i = 0; i < det_output.rows; i++)
	{

		cv::Mat classes_scores = det_output.row(i).colRange(4, numRows - keypoints_nums * 2); // 因为转置，numRows是此时的列
		cv::Point classIdPoint;
		double score;
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

		if (score > conf_threshold)
		{
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);
			int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
			int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
			int width = static_cast<int>(ow * x_factor);
			int height = static_cast<int>(oh * y_factor);
			std::vector<cv::Point2f> keypoints;

			Mat kpts = det_output.row(i).colRange(numRows - keypoints_nums * 2, numRows);
			for (int i = 0; i < keypoints_nums; i++)
			{
				float x = kpts.at<float>(0, i * 2 + 0) * x_factor;
				float y = kpts.at<float>(0, i * 2 + 1) * y_factor;
				// float s = kpts.at<float>(0, i * 3 + 2);
				keypoints.push_back(cv::Point2f(x, y));
			}
			boxes.push_back(cv::Rect(x, y, width, height));
			classIds.push_back(classIdPoint.x);
			confidences.push_back(score);
			objects_keypoints.push_back(keypoints);
		}
	}

	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);

	std::vector<Detection> detections{};

	for (unsigned long i = 0; i < nms_result.size(); ++i)
	{
		int idx = nms_result[i];

		Detection result;
		result.class_id = classIds[idx];

		// std::cout<<"class_id:"<<result.class_id<<std::endl;
		result.confidence = confidences[idx];
		// std::cout<<"-----------"<< result.class_id <<std::endl;

		result.className = classes[result.class_id];

		result.box = boxes[idx];

		result.key_points = objects_keypoints[idx];

		detections.push_back(result);
	}

	return detections;
};
