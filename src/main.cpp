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


#define MIN_CELL_ARC_LENGTH     5    // Min cell arc length
#define MAX_CELL_ARC_LENGTH     140  // Max cell arc length
#define DAPI_THRESHOLD          90   // DAPI enhancement threshold
#define GFP_THRESHOLD           20   // GFP  enhancement threshold
#define RFP_THRESHOLD           20   // RFP  enhancement threshold
#define COVERAGE_RATIO          0.3  // Coverage ratio
#define SOMA_FACTOR             1.5  // Soma radius = factor * nuclues radius
#define DEBUG_FLAG              1    // Debug flag


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
            cv::threshold(*normalized, *enhanced, DAPI_THRESHOLD, 255, cv::THRESH_BINARY);
        } break;

        case ChannelType::GFP: {
            // Create the mask
            cv::threshold(*normalized, *enhanced, GFP_THRESHOLD, 255, cv::THRESH_BINARY);
        } break;

        case ChannelType::RFP: {
            // Create the mask
            cv::threshold(*normalized, *enhanced, RFP_THRESHOLD, 255, cv::THRESH_BINARY);
        } break;

        default: {
            std::cerr << "Invalid channel type" << std::endl;
            return false;
        }
    }
    return true;
}

/* Find the contours in the image */
void contourCalc(cv::Mat src, double min_area, cv::Mat *dst, 
                    std::vector<std::vector<cv::Point>> *contours, 
                    std::vector<cv::Vec4i> *hierarchy, 
                    std::vector<HierarchyType> *validity_mask, 
                    std::vector<double> *parent_area) {

    cv::Mat temp_src;
    src.copyTo(temp_src);
    findContours(temp_src, *contours, *hierarchy, 
            cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
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

        // Eliminate small and large contours via contour arc calculation
        auto arc_length = arcLength(contours[i], true);
        if (    (arc_length >= MIN_CELL_ARC_LENGTH) && 
                (contours[i].size() >= 5) ) {
            filtered_contours->push_back(contours[i]);
        }
    }
}

/* Find cell soma */
bool findCellSoma( std::vector<cv::Point> nucleus_contour, 
                   cv::Mat cell_mask, 
                   std::vector<cv::Point> *soma_contour ) {

    bool status = false;

    // Calculate radius and center of the nucleus
    cv::Moments mu = moments(nucleus_contour, true);
    cv::Point2f mc = cv::Point2f(   static_cast<float>(mu.m10/mu.m00), 
                                    static_cast<float>(mu.m01/mu.m00)   );
    cv::RotatedRect min_area_rect = minAreaRect(cv::Mat(nucleus_contour));
    float radius = (float) sqrt(min_area_rect.size.width * min_area_rect.size.height);

    // Nucleus' region of influence
    cv::Mat roi_mask = cv::Mat::zeros(cell_mask.size(), CV_8UC1);
    float roi_radius = ((float) SOMA_FACTOR) * radius;
    cv::circle(roi_mask, mc, roi_radius, 255, -1, 8);
    std::vector<std::vector<cv::Point>> specific_contour (1, nucleus_contour);
    drawContours(   roi_mask, 
                    specific_contour, 
                    -1, 
                    cv::Scalar::all(0), 
                    cv::FILLED, 
                    cv::LINE_8, 
                    std::vector<cv::Vec4i>(), 
                    0, 
                    cv::Point()
                );
    int circle_score = countNonZero(roi_mask);

    // Soma present in ROI
    cv::Mat intersection;
    bitwise_and(roi_mask, cell_mask, intersection);
    int intersection_score = countNonZero(intersection);

    // Add to the soma mask if coverage area exceeds a certain threshold
    float ratio = ((float) intersection_score) / circle_score;
    if (ratio >= COVERAGE_RATIO) {

        // Segment
        cv::Mat soma_segmented;
        std::vector<std::vector<cv::Point>> contours_soma;
        std::vector<cv::Vec4i> hierarchy_soma;
        std::vector<HierarchyType> soma_contour_mask;
        std::vector<double> soma_contour_area;
        contourCalc(    intersection, 
                        1.0, 
                        &soma_segmented, 
                        &contours_soma, 
                        &hierarchy_soma, 
                        &soma_contour_mask, 
                        &soma_contour_area
                   );

        size_t max_index = 0;
        double max_area  = 0.0;
        for (size_t i = 0; i < contours_soma.size(); i++) {
            if (soma_contour_mask[i] != HierarchyType::PARENT_CNTR) continue;
            if (contours_soma[i].size() < 5) continue;
            auto arc_length = arcLength(contours_soma[i], true);
            if (arc_length >= MAX_CELL_ARC_LENGTH) continue;

            // Find the largest permissible contour
            if (soma_contour_area[i] > max_area) {
                max_area = soma_contour_area[i];
                max_index = i;
                status = true;
            }
        }
        *soma_contour = contours_soma[max_index];
    }
    return status;
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
        dia[i] = (float) 2 * sqrt(min_area_rect.size.width * min_area_rect.size.height);
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
    contourCalc(dapi_enhanced, 1.0, &dapi_segmented, &contours_dapi, 
                &hierarchy_dapi, &dapi_contour_mask, &dapi_contour_area);

    // Filter the gfp contours
    std::vector<std::vector<cv::Point>> contours_dapi_filtered;
    filterCells(contours_dapi, dapi_contour_mask, &contours_dapi_filtered);


    /* GFP image */
    // Enhance
    cv::Mat gfp_normalized, gfp_enhanced;
    if (!enhanceImage(  gfp, 
                        ChannelType::GFP, 
                        &gfp_normalized, 
                        &gfp_enhanced   )) {
        return false;
    }

    /* RFP image */
    // Enhance
    cv::Mat rfp_normalized, rfp_enhanced;
    if (!enhanceImage(  rfp, 
                        ChannelType::RFP, 
                        &rfp_normalized, 
                        &rfp_enhanced   )) {
        return false;
    }


    /** Collect the metrics **/

    std::vector<std::vector<cv::Point>> contours_gfp, contours_rfp;
    for (size_t i = 0; i < contours_dapi_filtered.size(); i++) {

        // Find DAPI-GFP Cell Soma
        std::vector<cv::Point> gfp_contour;
        if (findCellSoma( contours_dapi_filtered[i], gfp_enhanced, &gfp_contour )) {
            contours_gfp.push_back(gfp_contour);
        }

        // Find DAPI-RFP Cell Soma
        std::vector<cv::Point> rfp_contour;
        if (findCellSoma( contours_dapi_filtered[i], rfp_enhanced, &rfp_contour )) {
            contours_rfp.push_back(rfp_contour);
        }
    }

#if 0
    // Separation metrics for dapi-gfp cells
    float mean_dia = 0.0, stddev_dia = 0.0;
    float mean_aspect_ratio = 0.0, stddev_aspect_ratio = 0.0;
    separationMetrics(  dapi_gfp_contours, 
                        &mean_dia, 
                        &stddev_dia, 
                        &mean_aspect_ratio, 
                        &stddev_aspect_ratio
                     );
    data_stream << mean_dia << "," 
                << stddev_dia << "," 
                << mean_aspect_ratio << "," 
                << stddev_aspect_ratio << ",";

    // Separation metrics for other-gfp cells
    mean_dia = 0.0;
    stddev_dia = 0.0;
    mean_aspect_ratio = 0.0;
    stddev_aspect_ratio = 0.0;
    separationMetrics(  other_gfp_contours, 
                        &mean_dia, 
                        &stddev_dia, 
                        &mean_aspect_ratio, 
                        &stddev_aspect_ratio
                     );
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
                        &stddev_aspect_ratio
                     );
    data_stream << mean_dia << "," 
                << stddev_dia << "," 
                << mean_aspect_ratio << "," 
                << stddev_aspect_ratio << ",";

    // Separation metrics for other-rfp cells
    mean_dia = 0.0;
    stddev_dia = 0.0;
    mean_aspect_ratio = 0.0;
    stddev_aspect_ratio = 0.0;
    separationMetrics(  other_rfp_contours, 
                        &mean_dia, 
                        &stddev_dia, 
                        &mean_aspect_ratio, 
                        &stddev_aspect_ratio
                     );
    data_stream << mean_dia << "," 
                << stddev_dia << "," 
                << mean_aspect_ratio << "," 
                << stddev_aspect_ratio << ",";
#endif

    data_stream << std::endl;
    data_stream.close();


    /** Display the debug image **/

    if (DEBUG_FLAG) {
        // Initialize
        cv::Mat drawing_blue_debug  = dapi_enhanced;
        cv::Mat drawing_green_debug = gfp_enhanced;
        cv::Mat drawing_red_debug   = rfp_enhanced;

        // Merge the modified red, blue and green layers
        std::vector<cv::Mat> merge_debug;
        merge_debug.push_back(drawing_blue_debug);
        merge_debug.push_back(drawing_green_debug);
        merge_debug.push_back(drawing_red_debug);
        cv::Mat color_debug;
        cv::merge(merge_debug, color_debug);

        // Draw the debug image
        std::string out_debug = out_directory + analyzed_image_name;
        out_debug.insert(out_debug.find_last_of("."), "_debug", 6);
        cv::imwrite(out_debug.c_str(), color_debug);
    }


    /** Display the analyzed <dapi,gfp,rfp> image set **/

    // Initialize
    cv::Mat drawing_blue  = dapi_normalized;
    cv::Mat drawing_green = gfp_normalized;
    cv::Mat drawing_red   = rfp_normalized;

    // Draw GFP bondaries
    for (size_t i = 0; i < contours_gfp.size(); i++) {
        cv::RotatedRect min_ellipse = fitEllipse(cv::Mat(contours_gfp[i]));
        ellipse(drawing_blue, min_ellipse, 255, 2, 8);
        ellipse(drawing_green, min_ellipse, 255, 2, 8);
        ellipse(drawing_red, min_ellipse, 0, 2, 8);
    }

    // Draw RFP bondaries
    for (size_t i = 0; i < contours_rfp.size(); i++) {
        cv::RotatedRect min_ellipse = fitEllipse(cv::Mat(contours_rfp[i]));
        ellipse(drawing_blue, min_ellipse, 255, 2, 8);
        ellipse(drawing_green, min_ellipse, 0, 2, 8);
        ellipse(drawing_red, min_ellipse, 255, 2, 8);
    }

    // Merge the modified red, blue and green layers
    std::vector<cv::Mat> merge_analyzed;
    merge_analyzed.push_back(drawing_blue);
    merge_analyzed.push_back(drawing_green);
    merge_analyzed.push_back(drawing_red);
    cv::Mat color_analyzed;
    cv::merge(merge_analyzed, color_analyzed);

    // Draw the analyzed image
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

    data_stream << "Merged Image,";
    data_stream << "DAPI-GFP Cell Count,";
    data_stream << "Other-GFP Cell Count,";
    data_stream << "DAPI-RFP Cell Count,";
    data_stream << "Other-RFP Cell Count,";
    data_stream << "DAPI-GFP Cell Diameter (mean),";
    data_stream << "DAPI-GFP Cell Diameter (std. dev.),";
    data_stream << "DAPI-GFP Cell Aspect Ratio (mean),";
    data_stream << "DAPI-GFP Cell Aspect Ratio (std. dev.),";
    data_stream << "Other-GFP Cell Diameter (mean),";
    data_stream << "Other-GFP Cell Diameter (std. dev.),";
    data_stream << "Other-GFP Cell Aspect Ratio (mean),";
    data_stream << "Other-GFP Cell Aspect Ratio (std. dev.),";
    data_stream << "DAPI-RFP Cell Diameter (mean),";
    data_stream << "DAPI-RFP Cell Diameter (std. dev.),";
    data_stream << "DAPI-RFP Cell Aspect Ratio (mean),";
    data_stream << "DAPI-RFP Cell Aspect Ratio (std. dev.),";
    data_stream << "Other-RFP Cell Diameter (mean),";
    data_stream << "Other-RFP Cell Diameter (std. dev.),";
    data_stream << "Other-RFP Cell Aspect Ratio (mean),";
    data_stream << "Other-RFP Cell Aspect Ratio (std. dev.),";

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
