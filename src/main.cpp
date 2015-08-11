#include <iostream>
#include <dirent.h>
#include <sys/stat.h>
#include <fstream>
#include <cmath>
#include <cassert>
#include <string>
#include <iomanip>
#include <algorithm>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/photo/photo.hpp"


#define DAPI_MASK_RADIUS        12   // Dapi mask radius
#define DAPI_COVERAGE_RATIO     0.8  // Dapi coverage ratio
#define MIN_SOMA_SIZE           20   // Min soma size
#define MAX_ARC_LENGTH_FILTER   600  // Max arc length filter threshold
#define COVERAGE_RATIO          0.5  // Coverage ratio
#define SOMA_FACTOR             1.6  // Soma radius = factor * nuclues radius
#define PI                      3.14 // Approximate value of pi
#define NUM_AREA_BINS           11   // Number of bins
#define BIN_AREA                20   // Bin area
#define DEBUG_FLAG              0    // Debug flag
#define PLOTS_DIR_NAME          "plots/"                        // plots directory
#define SCATTERPLOT_CONTROL     "scatterplot_control.dat"       // Scatterplot control
#define SCATTERPLOT_SZ          "scatterplot_sz.dat"            // Scatterplot sz
#define SCATTER_CONTROL_STAT    "scatterplot_control_stat.dat"  // Scatterplot control stat
#define SCATTER_SZ_STAT         "scatterplot_sz_stat.dat"       // Scatterplot sz stat


/* Channel type */
enum class ChannelType : unsigned char {
    DAPI = 0,
    GFP,
    GFP_LOW,
    GFP_MEDIUM,
    GFP_HIGH,
    RFP_TYPE1,
    RFP_TYPE2
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
            cv::Mat temp1, temp2, temp3, temp4;
            // First threshold >= 85
            cv::threshold(*normalized, temp1, 85, 255, cv::THRESH_BINARY);
            // Second threshold >= 60, <= 70
            cv::threshold(*normalized, temp2, 70, 255, cv::THRESH_TOZERO_INV);
            cv::threshold(temp2, temp2, 60, 255, cv::THRESH_BINARY);
            // Third threshold >= 50, <= 45
            cv::threshold(*normalized, temp3, 50, 255, cv::THRESH_TOZERO_INV);
            cv::threshold(temp3, temp3, 45, 255, cv::THRESH_BINARY);
            // Fourth threshold >= 30, <= 25
            cv::threshold(*normalized, temp4, 30, 255, cv::THRESH_TOZERO_INV);
            cv::threshold(temp4, temp4, 25, 255, cv::THRESH_BINARY);

            // Merge the different thresholds
            cv::Mat tempor1, tempor2, tempor3;
            bitwise_or(temp1, temp2, tempor1);
            bitwise_or(tempor1, temp3, tempor2);
            bitwise_or(tempor2, temp4, tempor3);
            *enhanced = tempor3;
        } break;

        case ChannelType::GFP: {
            // Create the mask
            cv::threshold(*normalized, *enhanced, 25, 255, cv::THRESH_BINARY);
        } break;

        case ChannelType::GFP_LOW: {
            // Enhance the gfp low channel
            cv::threshold(*normalized, *enhanced, 25, 255, cv::THRESH_TOZERO);
            cv::threshold(*enhanced, *enhanced, 50, 255, cv::THRESH_TRUNC);
        } break;

        case ChannelType::GFP_MEDIUM: {
            // Enhance the gfp medium channel
            cv::threshold(*normalized, *enhanced, 50, 255, cv::THRESH_TOZERO);
            cv::threshold(*enhanced, *enhanced, 80, 255, cv::THRESH_TRUNC);
        } break;

        case ChannelType::GFP_HIGH: {
            // Enhance the gfp high channel
            cv::threshold(*normalized, *enhanced, 80, 255, cv::THRESH_BINARY);
        } break;

        case ChannelType::RFP_TYPE1: {
            // Create the mask
            cv::threshold(*normalized, *enhanced, 25, 255, cv::THRESH_BINARY);
        } break;

        case ChannelType::RFP_TYPE2: {
            // Create the mask
            cv::Mat temp1, temp2;
            // threshold >= 50
            cv::threshold(*normalized, temp1, 50, 255, cv::THRESH_BINARY);
            // 25 <= threshold <= 45
            cv::threshold(*normalized, temp2, 45, 255, cv::THRESH_TOZERO_INV);
            cv::threshold(temp2, temp2, 25, 255, cv::THRESH_BINARY);

            bitwise_or(temp1, temp2, *enhanced);
        } break;

        default: {
            std::cerr << "Invalid channel type" << std::endl;
            return false;
        }
    }
    return true;
}

/* Find the contours in the image */
void contourCalc(   cv::Mat src, ChannelType channel_type, 
                    double min_area, cv::Mat *dst, 
                    std::vector<std::vector<cv::Point>> *contours, 
                    std::vector<cv::Vec4i> *hierarchy, 
                    std::vector<HierarchyType> *validity_mask, 
                    std::vector<double> *parent_area    ) {

    cv::Mat temp_src;
    src.copyTo(temp_src);
    switch(channel_type) {
        case ChannelType::DAPI :
        case ChannelType::RFP_TYPE1 :
        case ChannelType::RFP_TYPE2 : {
            findContours(temp_src, *contours, *hierarchy, cv::RETR_EXTERNAL, 
                                                        cv::CHAIN_APPROX_SIMPLE);
        } break;

        case ChannelType::GFP :
        case ChannelType::GFP_LOW :
        case ChannelType::GFP_MEDIUM :
        case ChannelType::GFP_HIGH : {
            findContours(temp_src, *contours, *hierarchy, cv::RETR_CCOMP, 
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
void filterCells(   cv::Mat channel,
                    std::vector<std::vector<cv::Point>> contours,
                    std::vector<HierarchyType> contour_mask,
                    std::vector<std::vector<cv::Point>> *filtered_contours ) {

    for (size_t i = 0; i < contours.size(); i++) {
        if (contour_mask[i] != HierarchyType::PARENT_CNTR) continue;

        // Eliminate extremely small contours
        auto arc_length = arcLength(contours[i], true);
        if ((contours[i].size() >= 5) && (arc_length <= MAX_ARC_LENGTH_FILTER)) {

            // Calculate center of the nucleus
            cv::Moments mu = moments(contours[i], true);
            cv::Point2f mc = cv::Point2f(   static_cast<float>(mu.m10/mu.m00), 
                                            static_cast<float>(mu.m01/mu.m00)   );
            cv::Mat circle_mask = cv::Mat::zeros(channel.size(), CV_8UC1);
            cv::circle(circle_mask, mc, DAPI_MASK_RADIUS, 255, -1, 8);
            int circle_score = countNonZero(circle_mask);
            cv::Mat intersection;
            bitwise_or(circle_mask, channel, intersection);
            int intersection_score = countNonZero(intersection);

            // Add to the filter dapi list if coverage area exceeds a certain threshold
            float ratio = ((float) intersection_score) / circle_score;
            if (ratio >= DAPI_COVERAGE_RATIO) filtered_contours->push_back(contours[i]);
        }
    }
}

/* Find cell soma */
bool findCellSoma( std::vector<cv::Point> nucleus_contour, 
                   cv::Mat cell_mask, 
                   cv::Mat *intersection, 
                   std::vector<cv::Point> *soma_contour ) {

    bool status = false;

    // Calculate radius and center of the nucleus
    cv::Moments mu = moments(nucleus_contour, true);
    cv::Point2f mc = cv::Point2f(   static_cast<float>(mu.m10/mu.m00), 
                                    static_cast<float>(mu.m01/mu.m00)   );
    // Nucleus' region of influence
    cv::Mat roi_mask = cv::Mat::zeros(cell_mask.size(), CV_8UC1);
    float roi_radius = (float) (SOMA_FACTOR * DAPI_MASK_RADIUS);
    cv::circle(roi_mask, mc, roi_radius, 255, -1, 8);
    cv::circle(roi_mask, mc, DAPI_MASK_RADIUS, 0, -1, 8);
    int circle_score = countNonZero(roi_mask);

    // Soma present in ROI
    bitwise_and(roi_mask, cell_mask, *intersection);
    int intersection_score = countNonZero(*intersection);

    // Add the dapi contour to intersection region
    cv::circle(*intersection, mc, DAPI_MASK_RADIUS, 255, -1, 8);

    // Add to the soma mask if coverage area exceeds a certain threshold
    float ratio = ((float) intersection_score) / circle_score;
    if (ratio >= COVERAGE_RATIO) {

        // Segment
        cv::Mat soma_segmented;
        std::vector<std::vector<cv::Point>> contours_soma;
        std::vector<cv::Vec4i> hierarchy_soma;
        std::vector<HierarchyType> soma_contour_mask;
        std::vector<double> soma_contour_area;
        contourCalc(    *intersection, 
                        ChannelType::DAPI, 
                        1.0, 
                        &soma_segmented, 
                        &contours_soma, 
                        &hierarchy_soma, 
                        &soma_contour_mask, 
                        &soma_contour_area
                   );

        double max_area  = 0.0;
        for (size_t i = 0; i < contours_soma.size(); i++) {
            if (soma_contour_mask[i] != HierarchyType::PARENT_CNTR) continue;
            if (contours_soma[i].size() < 5) continue;
            if (soma_contour_area[i] < MIN_SOMA_SIZE) continue;

            // Find the largest permissible contour
            if (soma_contour_area[i] > max_area) {
                max_area = soma_contour_area[i];
                *soma_contour = contours_soma[i];
                status = true;
            }
        }
    }
    return status;
}

/* Separation metrics */
void separationMetrics( std::vector<std::vector<cv::Point>> contours, 
                        float *mean_diameter,
                        float *stddev_diameter,
                        float *mean_aspect_ratio,
                        float *stddev_aspect_ratio,
                        float *mean_error_ratio,
                        float *stddev_error_ratio ) {

    // Compute the normal distribution parameters of cells
    std::vector<cv::Point2f> mc(contours.size());
    std::vector<float> dia(contours.size());
    std::vector<float> aspect_ratio(contours.size());
    std::vector<float> error_ratio(contours.size());

    for (size_t i = 0; i < contours.size(); i++) {
        cv::Moments mu = moments(contours[i], true);
        mc[i] = cv::Point2f(static_cast<float>(mu.m10/mu.m00), 
                                            static_cast<float>(mu.m01/mu.m00));
        cv::RotatedRect min_area_rect = minAreaRect(cv::Mat(contours[i]));
        aspect_ratio[i] = float(min_area_rect.size.width)/min_area_rect.size.height;
        if (aspect_ratio[i] > 1.0) {
            aspect_ratio[i] = 1.0/aspect_ratio[i];
        }
        float actual_area = contourArea(contours[i]);
        dia[i] = 2 * sqrt(actual_area / PI);
        float ellipse_area = 
            (float) (PI * min_area_rect.size.width * min_area_rect.size.height);
        error_ratio[i] = (ellipse_area - actual_area) / ellipse_area;
    }
    cv::Scalar mean_dia, stddev_dia;
    cv::meanStdDev(dia, mean_dia, stddev_dia);
    *mean_diameter = static_cast<float>(mean_dia.val[0]);
    *stddev_diameter = static_cast<float>(stddev_dia.val[0]);

    cv::Scalar mean_ratio, stddev_ratio;
    cv::meanStdDev(aspect_ratio, mean_ratio, stddev_ratio);
    *mean_aspect_ratio = static_cast<float>(mean_ratio.val[0]);
    *stddev_aspect_ratio = static_cast<float>(stddev_ratio.val[0]);

    cv::Scalar mean_err_ratio, stddev_err_ratio;
    cv::meanStdDev(error_ratio, mean_err_ratio, stddev_err_ratio);
    *mean_error_ratio = static_cast<float>(mean_err_ratio.val[0]);
    *stddev_error_ratio = static_cast<float>(stddev_err_ratio.val[0]);
}

/* Group contour areas into bins */
void binArea(   std::vector<HierarchyType> contour_mask, 
                std::vector<double> contour_area, 
                std::string *contour_output ) {

    std::vector<unsigned int> count(NUM_AREA_BINS, 0);
    for (size_t i = 0; i < contour_mask.size(); i++) {
        if (contour_mask[i] != HierarchyType::PARENT_CNTR) continue;
        unsigned int area = static_cast<unsigned int>(round(contour_area[i]));
        unsigned int bin_index = 
            (area/BIN_AREA < NUM_AREA_BINS) ? area/BIN_AREA : NUM_AREA_BINS-1;
        count[bin_index]++;
    }

    unsigned int contour_cnt = 0;
    std::string area_binned;
    for (size_t i = 0; i < count.size(); i++) {
        area_binned += "," + std::to_string(count[i]);
        contour_cnt += count[i];
    }
    *contour_output = std::to_string(contour_cnt) + area_binned;
}

/* Scatter plot */
void scatterPlot(   std::string path,
                    bool is_control,
                    cv::Mat image,
                    std::vector<std::vector<cv::Point>> contours,
                    std::vector<HierarchyType> contour_mask     ) {

    std::string scatter_file = path;
    scatter_file += (is_control) ? SCATTERPLOT_CONTROL : SCATTERPLOT_SZ;
    std::ofstream data_stream;
    data_stream.open(scatter_file, std::ios::app);
    if (!data_stream.is_open()) {
        std::cerr << "Could not create the scatter file." << std::endl;
        return;
    }

    std::vector<float> avg_intensity(contours.size(), 0.f);
    cv::Mat labels = cv::Mat::zeros(image.size(), CV_8UC1);
    for (size_t i = 0; i < contours.size(); i++) {
        cv::drawContours(labels, contours, i, cv::Scalar(i+1), CV_FILLED);
    }
    std::vector<float> counts(contours.size(), 0.f);
    const int width = image.rows;
    for (size_t i = 0; i < (size_t) image.rows; i++) {
        for (size_t j = 0; j < (size_t) image.cols; j++) {
            unsigned char label = labels.data[i*width + j];
            if (!label) {
                continue;
            } else {
                label -= 1;
            }
            unsigned char value = image.data[i*width + j];
            avg_intensity[label] += value;
            ++counts[label];
        }
    }
    for (size_t i = 0; i < avg_intensity.size(); i++) {
        if (!counts[i] || !avg_intensity[i] || (counts[i] < 10) || (counts[i] > 1000)) continue;
        avg_intensity[i] /= counts[i];
        data_stream << std::setw(12) << (unsigned int) avg_intensity[i] 
                    << std::setw(10) << (unsigned int) counts[i] 
                    << std::endl;
    }
    data_stream.close();
}

/* Plot scatter plot mean and variance */
void scatterPlotStat( std::string path ) {

    for (unsigned int ctrl_indx = 0; ctrl_indx < 2; ctrl_indx++) {
        std::vector<unsigned int> plot[256];
        std::string scatter_file = path;
        scatter_file += (ctrl_indx) ? SCATTERPLOT_CONTROL : SCATTERPLOT_SZ;
        FILE *file = fopen(scatter_file.c_str(), "r");
        if (!file) {
            std::cerr << "Could not read scatterplot file." << std::endl;
            exit(1);
        }
        char line[128];
        while (fgets(line, sizeof(line), file) != NULL) {
            line[strlen(line)-1] = 0;
            char *str = strtok(line, " ");
            assert(str);
            auto intensity = atoi(str);
            str = strtok(NULL, " ");
            assert(str);
            auto area = (unsigned int) atoi(str);
            plot[intensity].insert(plot[intensity].begin(), area);
        }
        fclose(file);

        std::string stat_file = path;
        stat_file += (ctrl_indx) ? SCATTER_CONTROL_STAT : SCATTER_SZ_STAT;
        std::ofstream data_stream;
        data_stream.open(stat_file, std::ios::out);
        if (!data_stream.is_open()) {
            std::cerr << "Could not create the stat file." << std::endl;
            exit(1);
        }

        data_stream << std::setw(3) << "#x"
                    << std::setw(6) << "min"
                    << std::setw(6) << "q1"
                    << std::setw(6) << "median"
                    << std::setw(6) << "q3"
                    << std::setw(6) << "max"
                    << std::setw(6) << "width"
                    << std::setw(6) << "intensity"
                    << std::endl;
        unsigned int count = 0;
        for (unsigned int intensity = 0; intensity < 256; intensity++) {
            if (!plot[intensity].size()) continue;
            count++;
            std::sort(plot[intensity].begin(), plot[intensity].end());
            auto len = (unsigned int) plot[intensity].size();
            data_stream << std::setw(3) << count
                        << std::setw(6) << plot[intensity][0]
                        << std::setw(6) << plot[intensity][len/4]
                        << std::setw(6) << plot[intensity][len/2]
                        << std::setw(6) << plot[intensity][3*len/4]
                        << std::setw(6) << plot[intensity][len-1]
                        << std::setw(6) << "0.3"
                        << std::setw(6) << intensity
                        << std::endl;
        }
        data_stream.close();
    }
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
    contourCalc(dapi_enhanced, ChannelType::DAPI, 1.0, 
                &dapi_segmented, &contours_dapi, 
                &hierarchy_dapi, &dapi_contour_mask, 
                &dapi_contour_area);

    // Filter the dapi contours
    std::vector<std::vector<cv::Point>> contours_dapi_filtered;
    filterCells(dapi_enhanced, contours_dapi, dapi_contour_mask, &contours_dapi_filtered);


    /* GFP image */
    cv::Mat gfp_normalized, gfp_enhanced;
    if (!enhanceImage(  gfp, 
                        ChannelType::GFP, 
                        &gfp_normalized, 
                        &gfp_enhanced   )) {
        return false;
    }

    // GFP Low
    cv::Mat gfp_low_normalized, gfp_low_enhanced;
    if (!enhanceImage(  gfp, 
                        ChannelType::GFP_LOW, 
                        &gfp_low_normalized, 
                        &gfp_low_enhanced   )) {
        return false;
    }
    cv::Mat gfp_low_segmented;
    std::vector<std::vector<cv::Point>> contours_gfp_low;
    std::vector<cv::Vec4i> hierarchy_gfp_low;
    std::vector<HierarchyType> gfp_low_contour_mask;
    std::vector<double> gfp_low_contour_area;
    contourCalc(gfp_low_enhanced, ChannelType::GFP_LOW, 1.0, 
                &gfp_low_segmented, &contours_gfp_low, 
                &hierarchy_gfp_low, &gfp_low_contour_mask, 
                &gfp_low_contour_area);

    // GFP Medium
    cv::Mat gfp_medium_normalized, gfp_medium_enhanced;
    if (!enhanceImage(  gfp, 
                        ChannelType::GFP_MEDIUM, 
                        &gfp_medium_normalized, 
                        &gfp_medium_enhanced   )) {
        return false;
    }
    cv::Mat gfp_medium_segmented;
    std::vector<std::vector<cv::Point>> contours_gfp_medium;
    std::vector<cv::Vec4i> hierarchy_gfp_medium;
    std::vector<HierarchyType> gfp_medium_contour_mask;
    std::vector<double> gfp_medium_contour_area;
    contourCalc(gfp_medium_enhanced, ChannelType::GFP_MEDIUM, 1.0, 
                &gfp_medium_segmented, &contours_gfp_medium, 
                &hierarchy_gfp_medium, &gfp_medium_contour_mask, 
                &gfp_medium_contour_area);

    // GFP High
    cv::Mat gfp_high_normalized, gfp_high_enhanced;
    if (!enhanceImage(  gfp, 
                        ChannelType::GFP_HIGH, 
                        &gfp_high_normalized, 
                        &gfp_high_enhanced   )) {
        return false;
    }
    cv::Mat gfp_high_segmented;
    std::vector<std::vector<cv::Point>> contours_gfp_high;
    std::vector<cv::Vec4i> hierarchy_gfp_high;
    std::vector<HierarchyType> gfp_high_contour_mask;
    std::vector<double> gfp_high_contour_area;
    contourCalc(gfp_high_enhanced, ChannelType::GFP_HIGH, 1.0, 
                &gfp_high_segmented, &contours_gfp_high, 
                &hierarchy_gfp_high, &gfp_high_contour_mask, 
                &gfp_high_contour_area);

    /* RFP image */
    cv::Mat rfp_normalized, rfp_enhanced_type1;
    if (!enhanceImage(  rfp, 
                        ChannelType::RFP_TYPE1, 
                        &rfp_normalized, 
                        &rfp_enhanced_type1 )) {
        return false;
    }
    cv::Mat rfp_enhanced_type2;
    if (!enhanceImage(  rfp, 
                        ChannelType::RFP_TYPE2, 
                        &rfp_normalized, 
                        &rfp_enhanced_type2 )) {
        return false;
    }
    cv::Mat rfp_segmented;
    std::vector<std::vector<cv::Point>> contours_rfp_vec;
    std::vector<cv::Vec4i> hierarchy_rfp;
    std::vector<HierarchyType> rfp_contour_mask;
    std::vector<double> rfp_contour_area;
    contourCalc(rfp_enhanced_type2, ChannelType::RFP_TYPE2, 1.0, 
                &rfp_segmented, &contours_rfp_vec, 
                &hierarchy_rfp, &rfp_contour_mask, 
                &rfp_contour_area);

    /* RFP Scatter plot */
    // Determine whether the image is Control or SZ
    bool is_control = false;
    std::string sample_id = image_name.substr(0, 4);
    // Control - 3440, 3651, 4506, 9319, 9429, BJ2E, BJ3E
    if (    (sample_id == "3440") ||
            (sample_id == "3651") ||
            (sample_id == "4506") ||
            (sample_id == "9319") ||
            (sample_id == "9429") ||
            (sample_id == "BJ2E") ||
            (sample_id == "BJ3E")  ) {
        is_control = true;
    }
    // Note: Do nothing for SZ - 1792, 1835, 2038, 2497

    scatterPlot(    path + PLOTS_DIR_NAME,
                    is_control,
                    rfp_normalized,
                    contours_rfp_vec,
                    rfp_contour_mask    );


    /** Classify the cell soma **/

    std::vector<std::vector<cv::Point>> contours_gfp, contours_rfp;
    cv::Mat gfp_intersection = cv::Mat::zeros(gfp_enhanced.size(), CV_8UC1);
    cv::Mat rfp_intersection = cv::Mat::zeros(rfp_enhanced_type1.size(), CV_8UC1);
    for (size_t i = 0; i < contours_dapi_filtered.size(); i++) {

        // Find DAPI-GFP Cell Soma
        std::vector<cv::Point> gfp_contour;
        cv::Mat temp;
        if (findCellSoma( contours_dapi_filtered[i], gfp_enhanced, &temp, &gfp_contour )) {
            contours_gfp.push_back(gfp_contour);
            bitwise_or(gfp_intersection, temp, gfp_intersection);
            cv::Mat temp_not;
            bitwise_not(temp, temp_not);
            bitwise_and(gfp_enhanced, temp_not, gfp_enhanced);
        }

        // Find DAPI-RFP Cell Soma
        std::vector<cv::Point> rfp_contour;
        if (findCellSoma( contours_dapi_filtered[i], rfp_enhanced_type1, &temp, &rfp_contour )) {
            contours_rfp.push_back(rfp_contour);
            bitwise_or(rfp_intersection, temp, rfp_intersection);
            cv::Mat temp_not;
            bitwise_not(temp, temp_not);
            bitwise_and(rfp_enhanced_type1, temp_not, rfp_enhanced_type1);
        }
    }


    /** Collect the metrics **/

    // Separation metrics for dapi-gfp cells
    float mean_dia = 0.0, stddev_dia = 0.0;
    float mean_aspect_ratio = 0.0, stddev_aspect_ratio = 0.0;
    float mean_error_ratio = 0.0, stddev_error_ratio = 0.0;
    separationMetrics(  contours_gfp, 
                        &mean_dia, 
                        &stddev_dia, 
                        &mean_aspect_ratio, 
                        &stddev_aspect_ratio, 
                        &mean_error_ratio, 
                        &stddev_error_ratio
                     );
    data_stream << contours_gfp.size() << "," 
                << mean_dia << "," 
                << stddev_dia << "," 
                << mean_aspect_ratio << "," 
                << stddev_aspect_ratio << "," 
                << mean_error_ratio << "," 
                << stddev_error_ratio << ",";

    // Separation metrics for dapi-rfp cells
    mean_dia = 0.0;
    stddev_dia = 0.0;
    mean_aspect_ratio = 0.0;
    stddev_aspect_ratio = 0.0;
    mean_error_ratio = 0.0;
    stddev_error_ratio = 0.0;
    separationMetrics(  contours_rfp, 
                        &mean_dia, 
                        &stddev_dia, 
                        &mean_aspect_ratio, 
                        &stddev_aspect_ratio, 
                        &mean_error_ratio, 
                        &stddev_error_ratio
                     );
    data_stream << contours_rfp.size() << "," 
                << mean_dia << "," 
                << stddev_dia << "," 
                << mean_aspect_ratio << "," 
                << stddev_aspect_ratio << "," 
                << mean_error_ratio << "," 
                << stddev_error_ratio << ",";


    /* Characterize the gfp channel */
    // Gfp low
    std::string gfp_low_output;
    binArea(gfp_low_contour_mask, gfp_low_contour_area, &gfp_low_output);
    data_stream << gfp_low_output << ",";

    // Gfp medium
    std::string gfp_medium_output;
    binArea(gfp_medium_contour_mask, gfp_medium_contour_area, &gfp_medium_output);
    data_stream << gfp_medium_output << ",";

    // Gfp high
    std::string gfp_high_output;
    binArea(gfp_high_contour_mask, gfp_high_contour_area, &gfp_high_output);
    data_stream << gfp_high_output << ",";


    // End of entry
    data_stream << std::endl;
    data_stream.close();


    /** Display the debug image **/

    if (DEBUG_FLAG) {
        // Initialize
        cv::Mat drawing_blue_debug = cv::Mat::zeros(dapi_enhanced.size(), CV_8UC1);
        //cv::Mat drawing_blue_debug  = 2*dapi_normalized;
        //cv::Mat drawing_green_debug = cv::Mat::zeros(gfp_enhanced.size(), CV_8UC1);
        cv::Mat drawing_green_debug = gfp_intersection;
        cv::Mat drawing_red_debug   = cv::Mat::zeros(rfp_enhanced_type1.size(), CV_8UC1);
        //cv::Mat drawing_red_debug = rfp_intersection;

        // Draw DAPI bondaries
        for (size_t i = 0; i < contours_dapi_filtered.size(); i++) {
            cv::Moments mu = moments(contours_dapi_filtered[i], true);
            cv::Point2f mc = cv::Point2f(   static_cast<float>(mu.m10/mu.m00), 
                                            static_cast<float>(mu.m01/mu.m00)   );
            cv::circle(drawing_blue_debug, mc, DAPI_MASK_RADIUS, 255, 1, 8);
            cv::circle(drawing_green_debug, mc, DAPI_MASK_RADIUS, 255, 1, 8);
            cv::circle(drawing_red_debug, mc, DAPI_MASK_RADIUS, 255, 1, 8);
        }

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
    cv::Mat drawing_blue  = 2*dapi_normalized;
    cv::Mat drawing_green = gfp_normalized;
    cv::Mat drawing_red   = rfp_normalized;

    // Draw GFP bondaries
    for (size_t i = 0; i < contours_gfp.size(); i++) {
        //cv::RotatedRect min_ellipse = fitEllipse(cv::Mat(contours_gfp[i]));
        //ellipse(drawing_blue, min_ellipse, 255, 1, 8);
        //ellipse(drawing_green, min_ellipse, 255, 1, 8);
        //ellipse(drawing_red, min_ellipse, 0, 1, 8);
        drawContours(drawing_blue, contours_gfp, i, 255, 1, 8);
        drawContours(drawing_green, contours_gfp, i, 255, 1, 8);
        drawContours(drawing_red, contours_gfp, i, 0, 1, 8);
    }

    // Draw RFP bondaries
    for (size_t i = 0; i < contours_rfp.size(); i++) {
        //cv::RotatedRect min_ellipse = fitEllipse(cv::Mat(contours_rfp[i]));
        //ellipse(drawing_blue, min_ellipse, 255, 1, 8);
        //ellipse(drawing_green, min_ellipse, 0, 1, 8);
        //ellipse(drawing_red, min_ellipse, 255, 1, 8);
        drawContours(drawing_blue, contours_rfp, i, 255, 1, 8);
        drawContours(drawing_green, contours_rfp, i, 0, 1, 8);
        drawContours(drawing_red, contours_rfp, i, 255, 1, 8);
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

    /* Create the scatterplot Control and SZ file */
    // Create the plots directory
    std::string plots_directory = path + PLOTS_DIR_NAME;
    struct stat st = {0};
    if (stat(plots_directory.c_str(), &st) == -1) {
        mkdir(plots_directory.c_str(), 0700);
    }

    std::ofstream scatterplot_stream;

    // Control
    std::string scatter_control_file = plots_directory + SCATTERPLOT_CONTROL;
    scatterplot_stream.open(scatter_control_file, std::ios::out);
    if (!scatterplot_stream.is_open()) {
        std::cerr << "Could not create the scatter plot Control file." << std::endl;
        return -1;
    }
    scatterplot_stream.close();

    // SZ
    std::string scatter_sz_file = plots_directory + SCATTERPLOT_SZ;
    scatterplot_stream.open(scatter_sz_file, std::ios::out);
    if (!scatterplot_stream.is_open()) {
        std::cerr << "Could not create the scatter plot SZ file." << std::endl;
        return -1;
    }
    scatterplot_stream.close();

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
    data_stream << "DAPI-GFP Soma Diameter (mean),";
    data_stream << "DAPI-GFP Soma Diameter (std. dev.),";
    data_stream << "DAPI-GFP Soma Aspect Ratio (mean),";
    data_stream << "DAPI-GFP Soma Aspect Ratio (std. dev.),";
    data_stream << "DAPI-GFP Soma Error Ratio (mean),";
    data_stream << "DAPI-GFP Soma Error Ratio (std. dev.),";
    data_stream << "DAPI-RFP Cell Count,";
    data_stream << "DAPI-RFP Soma Diameter (mean),";
    data_stream << "DAPI-RFP Soma Diameter (std. dev.),";
    data_stream << "DAPI-RFP Soma Aspect Ratio (mean),";
    data_stream << "DAPI-RFP Soma Aspect Ratio (std. dev.),";
    data_stream << "DAPI-RFP Soma Error Ratio (mean),";
    data_stream << "DAPI-RFP Soma Error Ratio (std. dev.),";

    // GFP (low, medium and high) bins
    data_stream << "GFP_Low_Contour_Count,";
    for (unsigned int i = 0; i < NUM_AREA_BINS-1; i++) {
        data_stream << i*BIN_AREA << " <= GFP_Low_Contour_Area < " << (i+1)*BIN_AREA << ",";
    }
    data_stream << "GFP_Low_Contour_Area >= " << (NUM_AREA_BINS-1)*BIN_AREA << ",";

    data_stream << "GFP_Medium_Contour_Count,";
    for (unsigned int i = 0; i < NUM_AREA_BINS-1; i++) {
        data_stream << i*BIN_AREA << " <= GFP_Medium_Contour_Area < " << (i+1)*BIN_AREA << ",";
    }
    data_stream << "GFP_Medium_Contour_Area >= " << (NUM_AREA_BINS-1)*BIN_AREA << ",";

    data_stream << "GFP_High_Contour_Count,";
    for (unsigned int i = 0; i < NUM_AREA_BINS-1; i++) {
        data_stream << i*BIN_AREA << " <= GFP_High_Contour_Area < " << (i+1)*BIN_AREA << ",";
    }
    data_stream << "GFP_High_Contour_Area >= " << (NUM_AREA_BINS-1)*BIN_AREA << ",";


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

    /* Generate the scatter plot */
    std::cout << "Calculating the scatter plot statistics." << std::endl;
    scatterPlotStat(plots_directory);

    return 0;
}

