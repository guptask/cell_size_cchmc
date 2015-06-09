#include <iostream>
#include <dirent.h>
#include <sys/stat.h>
#include <fstream>
#include <cmath>
#include <cassert>
#include <string>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/photo/photo.hpp"


#define ROI_FACTOR              3    // ROI of microglial cell = roi factor * mean microglial dia
#define MIN_CELL_ARC_LENGTH     10   // Cell arc length
#define COVERAGE_RATIO          0.20 // Coverage Ratio

/* Channel type */
enum class ChannelType : unsigned char {
    DAPI = 0,
    GFP,
    RFP
};

/* Hierarchy type */
enum class HierarchyType : unsigned char {
    INVALID_CNTR = 0,
    CHILD_CNTR,
    PARENT_CNTR
};

/* Enhance the image */
bool enhanceImage(  cv::Mat src, 
                    ChannelType channel_type, 
                    cv::Mat *normalized, 
                    cv::Mat *enhanced       ) {

    // Split the image
    std::vector<cv::Mat> channel(3);
    cv::split(src, channel);
    cv::Mat img = channel[0];

    // Normalize the image
    cv::normalize(img, *normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // Enhance the image using Gaussian blur and thresholding
    switch (channel_type) {
        case ChannelType::DAPI: {
            // Create the mask
            cv::threshold(*normalized, *enhanced, 120, 255, cv::THRESH_BINARY);
        } break;

        case ChannelType::GFP: {
            // Sharpen the image
            cv::Mat temp;
            cv::GaussianBlur(*normalized, temp, cv::Size(0,0), 11);
            cv::addWeighted(*normalized, 1.5, temp, -0.5, 0, *enhanced);
            cv::threshold(*enhanced, *enhanced, 60, 255, cv::THRESH_BINARY);
        } break;

        case ChannelType::RFP: {
            // Sharpen the image
            cv::Mat temp;
            cv::GaussianBlur(*normalized, temp, cv::Size(0,0), 11);
            cv::addWeighted(*normalized, 1.5, temp, -0.5, 0, *enhanced);
            cv::threshold(*enhanced, *enhanced, 60, 255, cv::THRESH_BINARY);
        } break;

        default: {
            std::cerr << "Invalid channel type" << std::endl;
            return false;
        }
    }
    return true;
}

/* Find the contours in the image */
void contourCalc(cv::Mat src, ChannelType channel_type, 
                    double min_area, cv::Mat *dst, 
                    std::vector<std::vector<cv::Point>> *contours, 
                    std::vector<cv::Vec4i> *hierarchy, 
                    std::vector<HierarchyType> *validity_mask, 
                    std::vector<double> *parent_area) {

    cv::Mat temp_src;
    src.copyTo(temp_src);
    switch(channel_type) {
        case ChannelType::DAPI : {
            findContours(temp_src, *contours, *hierarchy, cv::RETR_EXTERNAL, 
                                                        cv::CHAIN_APPROX_SIMPLE);
        } break;

        default: return;
    }

    *dst = cv::Mat::zeros(temp_src.size(), CV_8UC3);
    if (!contours->size()) return;
    validity_mask->assign(contours->size(), HierarchyType::INVALID_CNTR);
    parent_area->assign(contours->size(), 0.0);

    // Keep the contours whose size is >= than min_area
    cv::RNG rng(12345);
    for (int index = 0 ; index < (int)contours->size(); index++) {
        if ((*hierarchy)[index][3] > -1) continue; // ignore child
        auto cntr_external = (*contours)[index];
        double area_external = fabs(contourArea(cv::Mat(cntr_external)));
        if (area_external < min_area) continue;

        std::vector<int> cntr_list;
        cntr_list.push_back(index);

        int index_hole = (*hierarchy)[index][2];
        double area_hole = 0.0;
        while (index_hole > -1) {
            std::vector<cv::Point> cntr_hole = (*contours)[index_hole];
            double temp_area_hole = fabs(contourArea(cv::Mat(cntr_hole)));
            if (temp_area_hole) {
                cntr_list.push_back(index_hole);
                area_hole += temp_area_hole;
            }
            index_hole = (*hierarchy)[index_hole][0];
        }
        double area_contour = area_external - area_hole;
        if (area_contour >= min_area) {
            (*validity_mask)[cntr_list[0]] = HierarchyType::PARENT_CNTR;
            (*parent_area)[cntr_list[0]] = area_contour;
            for (unsigned int i = 1; i < cntr_list.size(); i++) {
                (*validity_mask)[cntr_list[i]] = HierarchyType::CHILD_CNTR;
            }
            cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0,255), 
                                            rng.uniform(0,255));
            drawContours(*dst, *contours, index, color, CV_FILLED, 8, *hierarchy);
        }
    }
}

/* Filter out ill-formed or small cells */
void filterCells(   std::vector<std::vector<cv::Point>> contours,
                    std::vector<HierarchyType> contour_mask,
                    std::vector<std::vector<cv::Point>> *filtered_contours ) {

    for (size_t i = 0; i < contours.size(); i++) {
        if (contour_mask[i] != HierarchyType::PARENT_CNTR) continue;

        // Eliminate small contours via contour arc calculation
        if ((arcLength(contours[i], true) >= MIN_CELL_ARC_LENGTH) && 
                                            (contours[i].size() >= 5)) {
            filtered_contours->push_back(contours[i]);
        }
    }
}

/* Classify cells as dapi-gfp or dapi-rfp */
void classifyCells( std::vector<std::vector<cv::Point>> dapi_contours, 
                    cv::Mat dapi_gfp_intersection, 
                    cv::Mat dapi_rfp_intersection, 
                    std::vector<std::vector<cv::Point>> *dapi_gfp_contours,
                    std::vector<std::vector<cv::Point>> *dapi_rfp_contours,
                    std::vector<std::vector<cv::Point>> *dapi_other_contours    ) {

    for (size_t i = 0; i < dapi_contours.size(); i++) {

        std::vector<std::vector<cv::Point>> specific_contour (1, dapi_contours[i]);

        // Determine dapi-gfp score
        cv::Mat drawing_gfp = cv::Mat::zeros(dapi_gfp_intersection.size(), CV_8UC1);
        drawContours(   drawing_gfp, 
                        specific_contour, 
                        -1, 
                        cv::Scalar::all(255), 
                        cv::FILLED, 
                        cv::LINE_8, 
                        std::vector<cv::Vec4i>(), 
                        0, 
                        cv::Point()
                    );
        int dapi_non_zero_score = countNonZero(drawing_gfp);
        assert(dapi_non_zero_score);

        cv::Mat gfp_mask;
        bitwise_and(drawing_gfp, dapi_gfp_intersection, gfp_mask);
        int dapi_gfp_non_zero_score = countNonZero(gfp_mask);
        float dapi_gfp_coverage_ratio = 
                ((float)dapi_gfp_non_zero_score)/dapi_non_zero_score;

        // Determine dapi-rfp score
        cv::Mat drawing_rfp = cv::Mat::zeros(dapi_rfp_intersection.size(), CV_8UC1);
        drawContours(   drawing_rfp, 
                        specific_contour, 
                        -1, 
                        cv::Scalar::all(255), 
                        cv::FILLED, 
                        cv::LINE_8, 
                        std::vector<cv::Vec4i>(), 
                        0, 
                        cv::Point()
                    );

        cv::Mat rfp_mask;
        bitwise_and(drawing_rfp, dapi_rfp_intersection, rfp_mask);
        int dapi_rfp_non_zero_score = countNonZero(rfp_mask);
        float dapi_rfp_coverage_ratio = 
                ((float)dapi_rfp_non_zero_score)/dapi_non_zero_score;

        // Classify
        bool dapi_gfp_status = 
                (dapi_gfp_coverage_ratio > COVERAGE_RATIO) ? true : false;

        bool dapi_rfp_status = 
                (dapi_rfp_coverage_ratio > COVERAGE_RATIO) ? true : false;

        if (!dapi_gfp_status && !dapi_rfp_status) {
            dapi_other_contours->push_back(dapi_contours[i]);

        } else if (dapi_gfp_status && !dapi_rfp_status) {
            dapi_gfp_contours->push_back(dapi_contours[i]);

        } else if (!dapi_gfp_status && dapi_rfp_status) {
            dapi_rfp_contours->push_back(dapi_contours[i]);

        } else {
            if (dapi_gfp_coverage_ratio > dapi_rfp_coverage_ratio) {
                dapi_gfp_contours->push_back(dapi_contours[i]);

            } else if (dapi_gfp_coverage_ratio == dapi_rfp_coverage_ratio) {
                // may be assigned a separate class
                dapi_gfp_contours->push_back(dapi_contours[i]);

            } else {
                dapi_rfp_contours->push_back(dapi_contours[i]);
            }
        }
    }
}

/* Separation metrics */
void separationMetrics( std::vector<std::vector<cv::Point>> contours, 
                        float *mean_diameter,
                        float *stddev_diameter,
                        float *mean_aspect_ratio,
                        float *stddev_aspect_ratio  ) {

    // Compute the normal distribution parameters of cells
    std::vector<cv::Point2f> mc(contours.size());
    std::vector<float> dia(contours.size());
    std::vector<float> aspect_ratio(contours.size());

    for (size_t i = 0; i < contours.size(); i++) {
        cv::Moments mu = moments(contours[i], true);
        mc[i] = cv::Point2f(static_cast<float>(mu.m10/mu.m00), 
                                            static_cast<float>(mu.m01/mu.m00));
        cv::RotatedRect min_area_rect = minAreaRect(cv::Mat(contours[i]));
        dia[i] = (float) sqrt(pow(min_area_rect.size.width, 2) + 
                                                pow(min_area_rect.size.height, 2));
        aspect_ratio[i] = float(min_area_rect.size.width)/min_area_rect.size.height;
        if (aspect_ratio[i] > 1.0) {
            aspect_ratio[i] = 1.0/aspect_ratio[i];
        }
    }
    cv::Scalar mean_dia, stddev_dia;
    cv::meanStdDev(dia, mean_dia, stddev_dia);
    *mean_diameter = static_cast<float>(mean_dia.val[0]);
    *stddev_diameter = static_cast<float>(stddev_dia.val[0]);

    cv::Scalar mean_ratio, stddev_ratio;
    cv::meanStdDev(aspect_ratio, mean_ratio, stddev_ratio);
    *mean_aspect_ratio = static_cast<float>(mean_ratio.val[0]);
    *stddev_aspect_ratio = static_cast<float>(stddev_ratio.val[0]);
}

/* Process the images inside each directory */
bool processDir(std::string path, std::string image_name, std::string metrics_file) {

    /** Create the initial data collection skeleton **/

    // Open the metrics file
    std::ofstream data_stream;
    data_stream.open(metrics_file, std::ios::app);
    if (!data_stream.is_open()) {
        std::cerr << "Could not open the data output file." << std::endl;
        return false;
    }

    // Create the output image directory
    std::string out_directory = path + "result/";
    struct stat st = {0};
    if (stat(out_directory.c_str(), &st) == -1) {
        mkdir(out_directory.c_str(), 0700);
    }

    // Analyzed image name
    std::string analyzed_image_name = image_name;
    std::size_t found = analyzed_image_name.find("dapi");
    analyzed_image_name.replace(found, 4, "merge");
    data_stream << analyzed_image_name << ",";


    /** Extract the dapi, gfp and rfp streams for each input image **/

    // DAPI
    std::string in_filename = path + "jpg/" + image_name;
    cv::Mat dapi = cv::imread(in_filename.c_str(), -1);
    if (dapi.empty()) return false;

    // GFP
    std::string gfp_image_name = image_name;
    found = gfp_image_name.find("dapi");
    gfp_image_name.replace(found, 4, "gfp");
    in_filename = path + "jpg/" + gfp_image_name;
    cv::Mat gfp = cv::imread(in_filename.c_str(), -1);
    if (gfp.empty()) return false;

    // RFP
    std::string rfp_image_name = image_name;
    found = rfp_image_name.find("dapi");
    rfp_image_name.replace(found, 4, "rfp");
    in_filename = path + "jpg/" + rfp_image_name;
    cv::Mat rfp = cv::imread(in_filename.c_str(), -1);
    if (rfp.empty()) return false;


    /** Gather information needed for feature extraction **/

    /* DAPI image */
    // Enhance
    cv::Mat dapi_normalized, dapi_enhanced;
    if (!enhanceImage(  dapi, 
                        ChannelType::DAPI, 
                        &dapi_normalized, 
                        &dapi_enhanced  )) {
        return false;
    }

    // Segment
    cv::Mat dapi_segmented;
    std::vector<std::vector<cv::Point>> contours_dapi;
    std::vector<cv::Vec4i> hierarchy_dapi;
    std::vector<HierarchyType> dapi_contour_mask;
    std::vector<double> dapi_contour_area;
    contourCalc(dapi_enhanced, ChannelType::DAPI, 1.0, &dapi_segmented, 
        &contours_dapi, &hierarchy_dapi, &dapi_contour_mask, &dapi_contour_area);

    // Filter the dapi contours
    std::vector<std::vector<cv::Point>> contours_dapi_filtered;
    filterCells(contours_dapi, dapi_contour_mask, &contours_dapi_filtered);
    data_stream << contours_dapi_filtered.size() << ","; 

    /* GFP image */
    // Enhance
    cv::Mat gfp_normalized, gfp_enhanced;
    if (!enhanceImage(  gfp, 
                        ChannelType::GFP, 
                        &gfp_normalized, 
                        &gfp_enhanced   )) {
        return false;
    }

    // DAPI-GFP channel intersection
    cv::Mat dapi_gfp_intersection;
    bitwise_and(dapi_enhanced, gfp_enhanced, dapi_gfp_intersection);

    /* RFP image */
    // Enhance
    cv::Mat rfp_normalized, rfp_enhanced;
    if (!enhanceImage(  rfp, 
                        ChannelType::RFP, 
                        &rfp_normalized, 
                        &rfp_enhanced   )) {
        return false;
    }

    // DAPI-RFP channel intersection
    cv::Mat dapi_rfp_intersection;
    bitwise_and(dapi_enhanced, rfp_enhanced, dapi_rfp_intersection);


    /** Collect the metrics **/

    // Classify DAPI as DAPI-GFP, DAPI-RFP or DAPI-Other
    std::vector<std::vector<cv::Point>> dapi_gfp_contours, 
                                        dapi_rfp_contours, 
                                        dapi_other_contours;
    classifyCells(  contours_dapi_filtered, 
                    dapi_gfp_intersection, 
                    dapi_rfp_intersection, 
                    &dapi_gfp_contours, 
                    &dapi_rfp_contours,
                    &dapi_other_contours
                 );
    data_stream << dapi_gfp_contours.size() << "," 
                << dapi_rfp_contours.size() << ","
                << dapi_other_contours.size() << ",";

    // Separation metrics for dapi-gfp cells
    float mean_dia = 0.0, stddev_dia = 0.0;
    float mean_aspect_ratio = 0.0, stddev_aspect_ratio = 0.0;
    separationMetrics(  dapi_gfp_contours, 
                        &mean_dia, 
                        &stddev_dia, 
                        &mean_aspect_ratio, 
                        &stddev_aspect_ratio    );
    data_stream << mean_dia << "," 
                << stddev_dia << "," 
                << mean_aspect_ratio << "," 
                << stddev_aspect_ratio << ",";

    // Separation metrics for dapi-rfp cells
    mean_dia = 0.0;
    stddev_dia = 0.0;
    mean_aspect_ratio = 0.0;
    stddev_aspect_ratio = 0.0;
    separationMetrics(  dapi_rfp_contours, 
                        &mean_dia, 
                        &stddev_dia, 
                        &mean_aspect_ratio, 
                        &stddev_aspect_ratio    );
    data_stream << mean_dia << "," 
                << stddev_dia << "," 
                << mean_aspect_ratio << "," 
                << stddev_aspect_ratio << ",";

    // Separation metrics for dapi-other cells
    mean_dia = 0.0;
    stddev_dia = 0.0;
    mean_aspect_ratio = 0.0;
    stddev_aspect_ratio = 0.0;
    separationMetrics(  dapi_other_contours, 
                        &mean_dia, 
                        &stddev_dia, 
                        &mean_aspect_ratio, 
                        &stddev_aspect_ratio    );
    data_stream << mean_dia << "," 
                << stddev_dia << "," 
                << mean_aspect_ratio << "," 
                << stddev_aspect_ratio << ",";

    data_stream << std::endl;
    data_stream.close();


    /** Display the analyzed <dapi,gfp,rfp> image set **/

    // Initialize
    cv::Mat drawing_blue  = dapi_normalized;
    cv::Mat drawing_green = gfp_normalized;
    cv::Mat drawing_red   = rfp_normalized;

    // Draw DAPI-GFP boundaries
    for (size_t i = 0; i < dapi_gfp_contours.size(); i++) {
        cv::RotatedRect min_ellipse = fitEllipse(cv::Mat(dapi_gfp_contours[i]));
        ellipse(drawing_blue, min_ellipse, 255, 4, 8);
        ellipse(drawing_green, min_ellipse, 255, 4, 8);
        ellipse(drawing_red, min_ellipse, 255, 4, 8);
    }

    // Draw DAPI-RFP boundaries
    for (size_t i = 0; i < dapi_rfp_contours.size(); i++) {
        cv::RotatedRect min_ellipse = fitEllipse(cv::Mat(dapi_rfp_contours[i]));
        ellipse(drawing_blue, min_ellipse, 255, 4, 8);
        ellipse(drawing_green, min_ellipse, 255, 4, 8);
        ellipse(drawing_red, min_ellipse, 255, 4, 8);
    }

    // Draw DAPI-Other boundaries
    for (size_t i = 0; i < dapi_other_contours.size(); i++) {
        cv::RotatedRect min_ellipse = fitEllipse(cv::Mat(dapi_other_contours[i]));
        ellipse(drawing_blue, min_ellipse, 255, 4, 8);
        ellipse(drawing_green, min_ellipse, 255, 4, 8);
        ellipse(drawing_red, min_ellipse, 255, 4, 8);
    }

    // Merge the modified red, blue and green layers
    std::vector<cv::Mat> merge_analyzed;
    merge_analyzed.push_back(drawing_blue);
    merge_analyzed.push_back(drawing_green);
    merge_analyzed.push_back(drawing_red);
    cv::Mat color_analyzed;
    cv::merge(merge_analyzed, color_analyzed);
    std::string out_analyzed = out_directory + analyzed_image_name;
    cv::imwrite(out_analyzed.c_str(), color_analyzed);


    return true;
}

/* Main - create the threads and start the processing */
int main(int argc, char *argv[]) {

    /* Check for argument count */
    if (argc != 2) {
        std::cerr << "Invalid number of arguments." << std::endl;
        return -1;
    }

    /* Read the path to the data */
    std::string path(argv[1]);

    /* Read the list of directories to process */
    std::string image_list_filename = path + "image_list.dat";
    std::vector<std::string> input_images;
    FILE *file = fopen(image_list_filename.c_str(), "r");
    if (!file) {
        std::cerr << "Could not open 'image_list.dat' inside '" << path << "'." << std::endl;
        return -1;
    }
    char line[128];
    while (fgets(line, sizeof(line), file) != NULL) {
        line[strlen(line)-1] = 0;
        std::string temp_str(line);
        if (temp_str.find("dapi") == std::string::npos) continue;
        input_images.push_back(temp_str);
    }
    fclose(file);

    /* Create and prepare the file for metrics */
    std::string metrics_file_name = path.substr(path.find_first_of("/")+1);
    std::size_t found = metrics_file_name.find_first_of("/");
    assert(found != std::string::npos);
    metrics_file_name[found] = '_';
    found = metrics_file_name.find_first_of(" ");
    while (found != std::string::npos) {
        metrics_file_name[found] = '_';
        found = metrics_file_name.find_first_of(" ", found+1);
    }
    std::string metrics_file = path + "metrics_" + metrics_file_name + ".csv";
    std::ofstream data_stream;
    data_stream.open(metrics_file, std::ios::out);
    if (!data_stream.is_open()) {
        std::cerr << "Could not create the metrics file." << std::endl;
        return -1;
    }

    data_stream << "Merged Image,\
                    Total DAPI Cell Count,\
                    DAPI-GFP Cell Count,\
                    DAPI-RFP Cell Count,\
                    DAPI-Other Cell Count,\
                    DAPI-GFP Cell Diameter (mean),\
                    DAPI-GFP Cell Diameter (std. dev.),\
                    DAPI-GFP Cell Aspect Ratio (mean),\
                    DAPI-GFP Cell Aspect Ratio (std. dev.),\
                    DAPI-RFP Cell Diameter (mean),\
                    DAPI-RFP Cell Diameter (std. dev.),\
                    DAPI-RFP Cell Aspect Ratio (mean),\
                    DAPI-RFP Cell Aspect Ratio (std. dev.),\
                    DAPI-Other Cell Diameter (mean),\
                    DAPI-Other Cell Diameter (std. dev.),\
                    DAPI-Other Cell Aspect Ratio (mean),\
                    DAPI-Other Cell Aspect Ratio (std. dev.),\
                    ";

    data_stream << std::endl;
    data_stream.close();

    /* Process each image */
    for (unsigned int index = 0; index < input_images.size(); index++) {
        std::cout << "Processing " << input_images[index] << std::endl;
        if (!processDir(path, input_images[index], metrics_file)) {
            std::cout << "ERROR !!!" << std::endl;
            return -1;
        }
    }

    return 0;
}

